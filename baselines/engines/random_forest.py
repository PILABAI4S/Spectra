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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def reaction_prefix(reaction):
    return str(reaction).split("-")[0]

def reaction_suffix(reaction):
    return int(str(reaction).split("-")[1])

def condition_vector(record):
    return np.array(
        [
            float(record["Total_mols"]),
            float(record["Initiator"]),
            float(record["Time"]),
            float(record["Temp"]),
        ],
        dtype=np.float64,
    )

def property_vector(record):
    return np.array(
        [
            float(record.get("Mw") or 0.0),
            float(record.get("PDI") or 0.0),
            float(record.get("Conc_Mean") or 0.0),
        ],
        dtype=np.float64,
    )

def hit_count(record):
    mw, pdi, conc = property_vector(record)
    mw_hit = int(SUCCESS_MW_LOW <= mw <= SUCCESS_MW_HIGH)
    pdi_hit = int(pdi < SUCCESS_PDI_MAX)
    conc_hit = int(conc > SUCCESS_CONC_MIN)
    return mw_hit + pdi_hit + conc_hit, mw_hit, pdi_hit, conc_hit

def target_distance_values(mw, pdi, conc):
    mw_term = (mw - TARGET_MW) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_term = (pdi - TARGET_PDI) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_term = (conc - TARGET_CONC) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    return np.sqrt(mw_term**2 + pdi_term**2 + conc_term**2)

def window_penalty_values(mw, pdi, conc):
    mw_low = np.maximum(0.0, SUCCESS_MW_LOW - mw) / (TARGET_MW - SUCCESS_MW_LOW)
    mw_high = np.maximum(0.0, mw - SUCCESS_MW_HIGH) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_bad = np.maximum(0.0, pdi - SUCCESS_PDI_MAX) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_bad = np.maximum(0.0, SUCCESS_CONC_MIN - conc) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    return mw_low + mw_high + pdi_bad + conc_bad

def summarize_batch(batch_rows):
    enriched = []
    for row in batch_rows:
        hits, mw_hit, pdi_hit, conc_hit = hit_count(row)
        mw, pdi, conc = property_vector(row)
        enriched.append(
            {
                "row": row,
                "Target_hit_count": hits,
                "Mw_hit": mw_hit,
                "PDI_hit": pdi_hit,
                "Conc_Mean_hit": conc_hit,
                "Target_property_distance": float(target_distance_values(mw, pdi, conc)),
            }
        )
    best = sorted(enriched, key=lambda item: (-item["Target_hit_count"], item["Target_property_distance"]))[0]
    successes = [item for item in enriched if item["Target_hit_count"] == 3]
    return best, successes

def fit_model(current, seed):
    reactions = sorted(current)
    x = np.vstack([condition_vector(current[reaction]) for reaction in reactions])
    y = np.vstack([property_vector(current[reaction]) for reaction in reactions])
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    xz = x_scaler.fit_transform(x)
    yz = y_scaler.fit_transform(y)
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(xz, yz)
    pred_z = model.predict(xz)
    losses = {
        "Mw_Train_Loss_Z": float(mean_squared_error(yz[:, 0], pred_z[:, 0])),
        "PDI_Train_Loss_Z": float(mean_squared_error(yz[:, 1], pred_z[:, 1])),
        "Conc_Mean_Train_Loss_Z": float(mean_squared_error(yz[:, 2], pred_z[:, 2])),
    }
    return model, x_scaler, y_scaler, losses

def predict_with_uncertainty(model, x_scaler, y_scaler, rows):
    x = np.vstack([condition_vector(row) for row in rows])
    xz = x_scaler.transform(x)
    tree_preds_z = np.stack([tree.predict(xz) for tree in model.estimators_], axis=0)
    mean_z = tree_preds_z.mean(axis=0)
    std_z = tree_preds_z.std(axis=0)
    mean = y_scaler.inverse_transform(mean_z)
    std = std_z * y_scaler.scale_
    return mean, np.maximum(std, 1e-12)

def recommend(current, all_rows, all_by_batch, seed):
    model, x_scaler, y_scaler, losses = fit_model(current, seed)
    tested_batches = {row["Batch"] for row in current.values()}
    candidate_rows = [row for row in all_rows if row["Batch"] not in tested_batches]
    means, stds = predict_with_uncertainty(model, x_scaler, y_scaler, candidate_rows)
    mw_mu, pdi_mu, conc_mu = means[:, 0], means[:, 1], means[:, 2]
    distance = target_distance_values(mw_mu, pdi_mu, conc_mu)
    penalty = window_penalty_values(mw_mu, pdi_mu, conc_mu)
    uncertainty = stds[:, 0] / 5000.0 + stds[:, 1] / 0.5 + stds[:, 2] / 50.0
    score = penalty + 0.25 * distance - 0.05 * uncertainty
    best_idx = int(np.argmin(score))
    best_row = candidate_rows[best_idx]
    return {
        "best_row": best_row,
        "batch_rows": all_by_batch[best_row["Batch"]],
        "score": float(score[best_idx]),
        "predicted_target_distance": float(distance[best_idx]),
        "pred_mw": float(mw_mu[best_idx]),
        "pred_pdi": float(pdi_mu[best_idx]),
        "pred_conc": float(conc_mu[best_idx]),
        "std_mw": float(stds[best_idx, 0]),
        "std_pdi": float(stds[best_idx, 1]),
        "std_conc": float(stds[best_idx, 2]),
        "losses": losses,
        "candidate_count": len(candidate_rows),
    }

def run_one(seed, initial_rows, all_rows, all_by_reaction, all_by_batch):
    current = {row["Reaction"]: all_by_reaction[row["Reaction"]] for row in initial_rows}
    trajectory = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, MAX_ITERATIONS + 1):
        rec = recommend(current, all_rows, all_by_batch, seed + iteration)
        recommended = rec["best_row"]
        batch_rows = rec["batch_rows"]
        best, successes = summarize_batch(batch_rows)

        for row in batch_rows:
            current[row["Reaction"]] = row

        rec_hits, rec_mw_hit, rec_pdi_hit, rec_conc_hit = hit_count(recommended)
        best_row = best["row"]
        record = {
            "Seed": seed,
            "Iteration": iteration,
            "Recommended_Reaction": recommended["Reaction"],
            "Recommended_Batch": recommended["Batch"],
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
            "Batch_best_reaction": best_row["Reaction"],
            "Batch_best_Mw": best_row.get("Mw"),
            "Batch_best_PDI": best_row.get("PDI"),
            "Batch_best_Conc_Mean": best_row.get("Conc_Mean"),
            "Batch_best_target_hit_count": best["Target_hit_count"],
            "Batch_best_property_distance": best["Target_property_distance"],
            "Batch_success_count": len(successes),
            "Success_Reactions": ";".join(item["row"]["Reaction"] for item in successes),
            "RF_score": rec["score"],
            "Predicted_Target_Distance": rec["predicted_target_distance"],
            "Pred_Mw": rec["pred_mw"],
            "Pred_PDI": rec["pred_pdi"],
            "Pred_Conc_Mean": rec["pred_conc"],
            "Std_Mw": rec["std_mw"],
            "Std_PDI": rec["std_pdi"],
            "Std_Conc_Mean": rec["std_conc"],
            "Mw_Train_Loss_Z": rec["losses"]["Mw_Train_Loss_Z"],
            "PDI_Train_Loss_Z": rec["losses"]["PDI_Train_Loss_Z"],
            "Conc_Mean_Train_Loss_Z": rec["losses"]["Conc_Mean_Train_Loss_Z"],
            "Train_Size": len(current),
            "Candidate_Count": rec["candidate_count"],
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
