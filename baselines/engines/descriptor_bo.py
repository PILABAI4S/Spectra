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

from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge
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

def success(record):
    mw, pdi, conc = property_vector(record)
    return (
        SUCCESS_MW_LOW <= mw <= SUCCESS_MW_HIGH
        and pdi < SUCCESS_PDI_MAX
        and conc > SUCCESS_CONC_MIN
    )

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
    return window_penalty + PROPERTY_TARGET_WEIGHT * target_distance

class PropertyScalarBO:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.initial_rows = read_workbook(INITIAL_XLSX, "Reaction")
        self.all_rows = read_workbook(ALL_XLSX, "Reaction_ID")
        self.all_by_reaction = {row["Reaction"]: row for row in self.all_rows}
        self.all_by_batch = {}
        for row in self.all_rows:
            self.all_by_batch.setdefault(row["Batch"], []).append(row)
        for rows in self.all_by_batch.values():
            rows.sort(key=lambda r: r["Suffix"])
        self.current = {
            row["Reaction"]: self.all_by_reaction[row["Reaction"]]
            for row in self.initial_rows
        }

    def training_arrays(self):
        reactions = sorted(self.current)
        x = np.vstack([condition_vector(self.current[r]) for r in reactions])
        y = np.array(
            [math.log1p(property_objective(self.current[r])) for r in reactions],
            dtype=np.float64,
        )
        return reactions, x, y

    def fit_surrogate(self):
        _, x, y = self.training_arrays()
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        xz = x_scaler.fit_transform(x)
        yz = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        rff = RBFSampler(gamma=RFF_GAMMA, n_components=RFF_COMPONENTS, random_state=RFF_SEED)
        phi = rff.fit_transform(xz)
        model = BayesianRidge(max_iter=500, tol=1e-4, fit_intercept=True, compute_score=False)
        model.fit(phi, yz)
        pred_z = model.predict(phi)
        train_loss = float(np.mean((pred_z - yz) ** 2))
        return model, rff, x_scaler, y_scaler, train_loss

    def recommend(self, model, rff, x_scaler, y_scaler):
        tested_batches = {row["Batch"] for row in self.current.values()}
        candidate_rows = [row for row in self.all_rows if row["Batch"] not in tested_batches]
        x = np.vstack([condition_vector(row) for row in candidate_rows])
        phi = rff.transform(x_scaler.transform(x))
        mean_z, std_z = model.predict(phi, return_std=True)
        mean_log = y_scaler.inverse_transform(mean_z.reshape(-1, 1)).ravel()
        pred_objective = np.expm1(mean_log)
        _, _, train_y_log = self.training_arrays()
        best_y_z = float(np.min(y_scaler.transform(train_y_log.reshape(-1, 1)).ravel()))
        sigma = np.maximum(std_z, 1e-12)
        if ACQUISITION == "PI":
            z_score = (best_y_z - mean_z) / sigma
            acquisition = norm.cdf(z_score)
            best_idx = int(np.argmax(acquisition))
        elif ACQUISITION == "EI":
            improvement = best_y_z - mean_z - XI
            z_score = improvement / sigma
            acquisition = improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)
            best_idx = int(np.argmax(acquisition))
        elif ACQUISITION == "LCB":
            acquisition = mean_z - KAPPA * std_z
            best_idx = int(np.argmin(acquisition))
        elif ACQUISITION == "UCB":
            # Upper confidence bound on a loss: prefer candidates whose
            # pessimistic objective estimate remains small.
            acquisition = mean_z + KAPPA * std_z
            best_idx = int(np.argmin(acquisition))
        else:
            acquisition = -pred_objective
            best_idx = int(np.argmin(pred_objective))
        best_row = candidate_rows[best_idx]
        batch_rows = self.all_by_batch[best_row["Batch"]]
        success_rows = [row for row in batch_rows if success(row)]
        nearest = sorted(
            batch_rows,
            key=lambda row: (
                0 if success(row) else 1,
                abs(float(row.get("Mw") or 0) - TARGET_MW),
                max(0.0, float(row.get("PDI") or 0) - SUCCESS_PDI_MAX),
                max(0.0, SUCCESS_CONC_MIN - float(row.get("Conc_Mean") or 0)),
            ),
        )[0]
        return {
            "best_row": best_row,
            "batch_rows": batch_rows,
            "success_rows": success_rows,
            "nearest_row": nearest,
            "predicted_objective": float(pred_objective[best_idx]),
            "predicted_log_objective": float(mean_log[best_idx]),
            "predicted_std_z": float(std_z[best_idx]),
            "acquisition_value": float(acquisition[best_idx]),
            "candidate_count": len(candidate_rows),
        }

    def add_batch(self, batch_rows):
        for row in batch_rows:
            self.current[row["Reaction"]] = row

    def run(self):
        logs = []
        stop_reason = "max_iterations"
        for iteration in range(1, MAX_ITERATIONS + 1):
            model, rff, x_scaler, y_scaler, train_loss = self.fit_surrogate()
            rec = self.recommend(model, rff, x_scaler, y_scaler)
            best = rec["best_row"]
            nearest = rec["nearest_row"]
            log = {
                "Iteration": iteration,
                "Recommended_Reaction": best["Reaction"],
                "Recommended_Batch": best["Batch"],
                "Predicted_Property_Objective": rec["predicted_objective"],
                "Predicted_Log_Objective": rec["predicted_log_objective"],
                "Predicted_Std_Z": rec["predicted_std_z"],
                "Acquisition": f"{ACQUISITION}(kappa={KAPPA}, xi={XI})",
                "Acquisition_Value": rec["acquisition_value"],
                "Train_Loss_Z": train_loss,
                "Train_Size": len(self.current),
                "Candidate_Count": rec["candidate_count"],
                "Batch_Success_Count": len(rec["success_rows"]),
                "Success_Reactions": ";".join(row["Reaction"] for row in rec["success_rows"]),
                "Nearest_In_Batch": nearest["Reaction"],
                "Nearest_Mw": nearest["Mw"],
                "Nearest_PDI": nearest["PDI"],
                "Nearest_Conc_Mean": nearest["Conc_Mean"],
                "Model": f"BayesianRidge + RBFSampler(n_components={RFF_COMPONENTS}, gamma={RFF_GAMMA})",
            }
            logs.append(log)
            self.write_outputs(logs, stop_reason)
            if rec["success_rows"]:
                stop_reason = "success_condition_met"
                self.write_outputs(logs, stop_reason)
                break
            self.add_batch(rec["batch_rows"])
        return logs, stop_reason

    def write_outputs(self, logs, stop_reason):
        if logs:
            with (OUT_DIR / "bo_property_scalar_iteration_log.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
                writer.writeheader()
                writer.writerows(logs)
        summary = {
            "method": "Property scalar Bayesian optimization using the same acquisition family as spectral-distance BO",
            "input": ["Total_mols", "Initiator", "Time", "Temp"],
            "output": "log1p(property window penalty + target-center distance)",
            "acquisition": ACQUISITION,
            "kappa": KAPPA,
            "xi": XI,
            "property_target_weight": PROPERTY_TARGET_WEIGHT,
            "rff_seed": RFF_SEED,
            "max_iterations": MAX_ITERATIONS,
            "success_criteria": {
                "Mw": [SUCCESS_MW_LOW, SUCCESS_MW_HIGH],
                "PDI": f"<{SUCCESS_PDI_MAX}",
                "Conc_Mean": f">{SUCCESS_CONC_MIN}",
            },
            "stop_reason": stop_reason,
            "iterations": len(logs),
            "final": logs[-1] if logs else None,
        }
        with (OUT_DIR / "bo_property_scalar_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
