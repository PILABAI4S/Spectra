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

def hit_count(record):
    mw = as_float(record, "Mw")
    pdi = as_float(record, "PDI")
    conc = as_float(record, "Conc_Mean")
    mw_hit = int(SUCCESS_MW_LOW <= mw <= SUCCESS_MW_HIGH)
    pdi_hit = int(pdi < SUCCESS_PDI_MAX)
    conc_hit = int(conc > SUCCESS_CONC_MIN)
    return mw_hit + pdi_hit + conc_hit, mw_hit, pdi_hit, conc_hit

def success(record):
    return hit_count(record)[0] == 3

def target_distance(record):
    mw = as_float(record, "Mw")
    pdi = as_float(record, "PDI")
    conc = as_float(record, "Conc_Mean")
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
            }
        )
    best = sorted(enriched, key=lambda item: (-item["Target_hit_count"], item["Target_property_distance"]))[0]
    successes = [item for item in enriched if item["Target_hit_count"] == 3]
    return best, successes

def run_one(seed, initial_rows, all_rows, all_by_reaction, all_by_batch):
    rng = np.random.default_rng(seed)
    current = {row["Reaction"]: all_by_reaction[row["Reaction"]] for row in initial_rows}
    tested_batches = {row["Batch"] for row in current.values()}
    trajectory = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, MAX_ITERATIONS + 1):
        candidate_rows = [row for row in all_rows if row["Batch"] not in tested_batches]
        if not candidate_rows:
            stop_reason = "candidate_space_exhausted"
            break

        recommended = candidate_rows[int(rng.integers(0, len(candidate_rows)))]
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
            "Batch_best_reaction": best_row["Reaction"],
            "Batch_best_Mw": best_row.get("Mw"),
            "Batch_best_PDI": best_row.get("PDI"),
            "Batch_best_Conc_Mean": best_row.get("Conc_Mean"),
            "Batch_best_target_hit_count": best["Target_hit_count"],
            "Batch_best_property_distance": best["Target_property_distance"],
            "Batch_success_count": len(successes),
            "Success_Reactions": ";".join(item["row"]["Reaction"] for item in successes),
            "Train_Size": len(current),
            "Remaining_Candidates": max(len(candidate_rows) - len(batch_rows), 0),
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
