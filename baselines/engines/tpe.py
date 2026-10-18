"""Portable engine; numerical definitions retained from the executed baseline."""

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from benchmark.data import read_workbook
from baselines.runtime import engine_settings

if "RUNTIME_CONFIG" not in globals():
    raise RuntimeError("Use run_benchmark.py to configure this engine")
globals().update(engine_settings(RUNTIME_CONFIG))



def reaction_prefix(reaction):
    return str(reaction).split("-")[0]

def reaction_suffix(reaction):
    return int(str(reaction).split("-")[1])

def as_float(record, key):
    value = record.get(key)
    return float(value) if value is not None else np.nan

def condition_vector(record):
    return np.array(
        [
            as_float(record, "Total_mols"),
            as_float(record, "Initiator"),
            as_float(record, "Time"),
            as_float(record, "Temp"),
        ],
        dtype=np.float64,
    )

def property_vector(record):
    return np.array(
        [as_float(record, "Mw"), as_float(record, "PDI"), as_float(record, "Conc_Mean")],
        dtype=np.float64,
    )

def hit_count(record):
    mw, pdi, conc = property_vector(record)
    mw_hit = int(SUCCESS_MW_LOW <= mw <= SUCCESS_MW_HIGH)
    pdi_hit = int(pdi < SUCCESS_PDI_MAX)
    conc_hit = int(conc > SUCCESS_CONC_MIN)
    return mw_hit + pdi_hit + conc_hit, mw_hit, pdi_hit, conc_hit

def success(record):
    return hit_count(record)[0] == 3

def property_objective(record):
    mw, pdi, conc = property_vector(record)
    mw_low = max(0.0, SUCCESS_MW_LOW - mw) / (TARGET_MW - SUCCESS_MW_LOW)
    mw_high = max(0.0, mw - SUCCESS_MW_HIGH) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_bad = max(0.0, pdi - SUCCESS_PDI_MAX) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_bad = max(0.0, SUCCESS_CONC_MIN - conc) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    window_penalty = mw_low + mw_high + pdi_bad + conc_bad
    mw_term = (mw - TARGET_MW) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_term = (pdi - TARGET_PDI) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_term = (conc - TARGET_CONC) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    target_distance = math.sqrt(mw_term**2 + pdi_term**2 + conc_term**2)
    return float(window_penalty + 0.25 * target_distance)

def target_distance(record):
    mw, pdi, conc = property_vector(record)
    mw_term = (mw - TARGET_MW) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_term = (pdi - TARGET_PDI) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_term = (conc - TARGET_CONC) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    return float(np.sqrt(mw_term**2 + pdi_term**2 + conc_term**2))

def summarize_batch(batch_rows):
    enriched = []
    for row in batch_rows:
        hits, mw_hit, pdi_hit, conc_hit = hit_count(row)
        enriched.append(
            {
                "row": row,
                "Target_hit_count": hits,
                "Mw_hit": mw_hit,
                "PDI_hit": pdi_hit,
                "Conc_Mean_hit": conc_hit,
                "Target_property_distance": target_distance(row),
                "Property_objective": property_objective(row),
            }
        )
    best = sorted(
        enriched,
        key=lambda item: (
            -item["Target_hit_count"],
            item["Property_objective"],
            item["Target_property_distance"],
        ),
    )[0]
    successes = [item for item in enriched if item["Target_hit_count"] == 3]
    return best, successes

def log_gaussian_mixture_density(x, samples, bandwidth):
    if len(samples) == 0:
        return np.full(len(x), -1e12, dtype=np.float64)
    diff = (x[:, None, :] - samples[None, :, :]) / bandwidth.reshape(1, 1, -1)
    log_comp = -0.5 * np.sum(diff * diff, axis=2) - np.sum(np.log(bandwidth)) - 0.5 * x.shape[1] * math.log(2.0 * math.pi)
    max_log = np.max(log_comp, axis=1)
    return max_log + np.log(np.mean(np.exp(log_comp - max_log[:, None]), axis=1))

def tpe_recommend(rng, current_rows, candidate_batches, all_by_batch):
    train_rows = list(current_rows.values())
    x_train_raw = np.vstack([condition_vector(row) for row in train_rows])
    y_train = np.array([property_objective(row) for row in train_rows], dtype=np.float64)

    center = np.nanmean(x_train_raw, axis=0)
    scale = np.nanstd(x_train_raw, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    x_train = (x_train_raw - center) / scale

    n_good = max(2, int(math.ceil(len(train_rows) * GAMMA_QUANTILE)))
    order = np.argsort(y_train)
    good = x_train[order[:n_good]]
    bad = x_train[order[n_good:]]
    if len(bad) < 2:
        bad = x_train[order[-min(len(order), n_good):]]

    bandwidth = np.nanstd(x_train, axis=0) * (len(train_rows) ** (-1.0 / 5.0))
    bandwidth = np.where(bandwidth < 0.15, 0.15, bandwidth)

    batches = list(candidate_batches)
    if len(batches) > N_EI_CANDIDATES:
        sampled_batches = rng.choice(batches, size=N_EI_CANDIDATES, replace=False).tolist()
    else:
        sampled_batches = batches

    representatives = [all_by_batch[batch][int(rng.integers(0, len(all_by_batch[batch])))] for batch in sampled_batches]
    if rng.random() < RANDOM_PROB or len(train_rows) <= N_STARTUP_ITERATIONS * 24:
        idx = int(rng.integers(0, len(representatives)))
        row = representatives[idx]
        return row, {
            "TPE_Log_L_over_G": np.nan,
            "TPE_Log_Good": np.nan,
            "TPE_Log_Bad": np.nan,
            "TPE_Mode": "random_startup_or_exploration",
            "TPE_Candidate_Count": len(representatives),
            "TPE_Good_Count": len(good),
            "TPE_Bad_Count": len(bad),
        }

    x_cand = np.vstack([condition_vector(row) for row in representatives])
    x_cand = (x_cand - center) / scale
    log_good = log_gaussian_mixture_density(x_cand, good, bandwidth)
    log_bad = log_gaussian_mixture_density(x_cand, bad, bandwidth)
    score = log_good - log_bad
    idx = int(np.argmax(score))
    row = representatives[idx]
    return row, {
        "TPE_Log_L_over_G": float(score[idx]),
        "TPE_Log_Good": float(log_good[idx]),
        "TPE_Log_Bad": float(log_bad[idx]),
        "TPE_Mode": "density_ratio",
        "TPE_Candidate_Count": len(representatives),
        "TPE_Good_Count": len(good),
        "TPE_Bad_Count": len(bad),
    }

def run_one(seed, initial_rows, all_rows, all_by_reaction, all_by_batch):
    rng = np.random.default_rng(seed)
    current = {row["Reaction"]: all_by_reaction[row["Reaction"]] for row in initial_rows}
    tested_batches = {row["Batch"] for row in current.values()}
    trajectory = []
    stop_reason = "max_iterations_reached"

    all_batches = sorted(all_by_batch)
    for iteration in range(1, MAX_ITERATIONS + 1):
        candidate_batches = [batch for batch in all_batches if batch not in tested_batches]
        if not candidate_batches:
            stop_reason = "candidate_space_exhausted"
            break

        recommended, tpe_info = tpe_recommend(rng, current, candidate_batches, all_by_batch)
        batch = recommended["Batch"]
        batch_rows = all_by_batch[batch]
        best, successes = summarize_batch(batch_rows)
        for row in batch_rows:
            current[row["Reaction"]] = row
        tested_batches.add(batch)

        rec_hits, rec_mw_hit, rec_pdi_hit, rec_conc_hit = hit_count(recommended)
        best_row = best["row"]
        record = {
            "Seed": seed,
            "Iteration": iteration,
            "Recommended_Reaction": recommended["Reaction"],
            "Recommended_Batch": batch,
            "Total_mols": recommended.get("Total_mols"),
            "Initiator": recommended.get("Initiator"),
            "Time": recommended.get("Time"),
            "Temp": recommended.get("Temp"),
            "Mw": recommended.get("Mw"),
            "PDI": recommended.get("PDI"),
            "Conc_Mean": recommended.get("Conc_Mean"),
            "Target_hit_count": rec_hits,
            "Mw_hit": rec_mw_hit,
            "PDI_hit": rec_pdi_hit,
            "Conc_Mean_hit": rec_conc_hit,
            "Property_objective": property_objective(recommended),
            "Target_property_distance": target_distance(recommended),
            "Batch_best_reaction": best_row["Reaction"],
            "Batch_best_Mw": best_row.get("Mw"),
            "Batch_best_PDI": best_row.get("PDI"),
            "Batch_best_Conc_Mean": best_row.get("Conc_Mean"),
            "Batch_best_target_hit_count": best["Target_hit_count"],
            "Batch_best_property_objective": best["Property_objective"],
            "Batch_best_property_distance": best["Target_property_distance"],
            "Batch_success_count": len(successes),
            "Success_Reactions": ";".join(item["row"]["Reaction"] for item in successes),
            "Train_Size": len(current),
            "Remaining_Batches": len(candidate_batches) - 1,
            **tpe_info,
        }
        trajectory.append(record)
        if successes:
            stop_reason = "success_condition_met"
            break

    final = trajectory[-1] if trajectory else {}
    return {
        "Seed": seed,
        "Stop_reason": stop_reason,
        "Success": stop_reason == "success_condition_met",
        "Iterations": int(final.get("Iteration", 0)),
        "Recommended_Reaction": final.get("Recommended_Reaction"),
        "Recommended_Batch": final.get("Recommended_Batch"),
        "Batch_best_reaction": final.get("Batch_best_reaction"),
        "Batch_success_count": final.get("Batch_success_count", 0),
        "Success_Reactions": final.get("Success_Reactions", ""),
    }, trajectory
