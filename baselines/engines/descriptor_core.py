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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        dtype=np.float32,
    )

def property_vector(record):
    return np.array(
        [as_float(record, "Mw"), as_float(record, "PDI"), as_float(record, "Conc_Mean")],
        dtype=np.float32,
    )

def hit_count(record):
    mw, pdi, conc = property_vector(record)
    mw_hit = int(SUCCESS_MW_LOW <= mw <= SUCCESS_MW_HIGH)
    pdi_hit = int(pdi < SUCCESS_PDI_MAX)
    conc_hit = int(conc > SUCCESS_CONC_MIN)
    return mw_hit + pdi_hit + conc_hit, mw_hit, pdi_hit, conc_hit

def property_score_from_values(mw, pdi, conc):
    mw_low = max(0.0, SUCCESS_MW_LOW - mw) / (TARGET_MW - SUCCESS_MW_LOW)
    mw_high = max(0.0, mw - SUCCESS_MW_HIGH) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_bad = max(0.0, pdi - SUCCESS_PDI_MAX) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_bad = max(0.0, SUCCESS_CONC_MIN - conc) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    window_penalty = mw_low + mw_high + pdi_bad + conc_bad
    mw_term = (mw - TARGET_MW) / (SUCCESS_MW_HIGH - TARGET_MW)
    pdi_term = (pdi - TARGET_PDI) / max(SUCCESS_PDI_MAX - TARGET_PDI, 1e-6)
    conc_term = (conc - TARGET_CONC) / max(TARGET_CONC - SUCCESS_CONC_MIN, 1e-6)
    target_pull = math.sqrt(mw_term**2 + pdi_term**2 + conc_term**2)
    return float(window_penalty + 0.25 * target_pull)

def property_score(record):
    mw, pdi, conc = property_vector(record)
    return property_score_from_values(float(mw), float(pdi), float(conc))

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
                "Property_score": property_score(row),
            }
        )
    best = sorted(
        enriched,
        key=lambda item: (
            -item["Target_hit_count"],
            item["Property_score"],
            item["Target_property_distance"],
        ),
    )[0]
    successes = [item for item in enriched if item["Target_hit_count"] == 3]
    return best, successes

class ConditionOnlyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(1, D_MODEL)
        self.feature_embed = nn.Parameter(torch.randn(4, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=D_MODEL * FF_MULT,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 3),
        )

    def forward(self, x):
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))

def make_arrays(rows):
    x = np.vstack([condition_vector(row) for row in rows]).astype(np.float32)
    y = np.vstack([property_vector(row) for row in rows]).astype(np.float32)
    return x, y

def normalize(train_x, train_y):
    x_center = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_center = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    return x_center, x_scale, y_center, y_scale

def weighted_mse(pred, target):
    weights = torch.tensor([MW_WEIGHT, PDI_WEIGHT, CONC_WEIGHT], dtype=pred.dtype, device=pred.device)
    return torch.mean(((pred - target) ** 2) * weights)

def train_model(train_rows, seed):
    set_seed(seed)
    x, y = make_arrays(train_rows)
    x_center, x_scale, y_center, y_scale = normalize(x, y)
    xz = (x - x_center) / x_scale
    yz = (y - y_center) / y_scale

    valid_mask = np.array([row["Suffix"] in {1, 5, 10, 15, 20} for row in train_rows], dtype=bool)
    if valid_mask.sum() >= 4 and (~valid_mask).sum() >= 8:
        train_mask = ~valid_mask
    else:
        train_mask = np.ones(len(train_rows), dtype=bool)
        valid_mask = np.zeros(len(train_rows), dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionOnlyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_ds = TensorDataset(torch.tensor(xz[train_mask]), torch.tensor(yz[train_mask]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True, generator=generator)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_pred = model(torch.tensor(xz, dtype=torch.float32, device=device)).cpu()
        all_target = torch.tensor(yz, dtype=torch.float32)
        train_loss = float(weighted_mse(all_pred[train_mask], all_target[train_mask]).item())
        valid_loss = float(weighted_mse(all_pred[valid_mask], all_target[valid_mask]).item()) if valid_mask.any() else np.nan
    return model, {"x_center": x_center, "x_scale": x_scale, "y_center": y_center, "y_scale": y_scale}, train_loss, valid_loss, device

def predict_rows(model, norm, rows, device):
    x = np.vstack([condition_vector(row) for row in rows]).astype(np.float32)
    xz = (x - norm["x_center"]) / norm["x_scale"]
    with torch.no_grad():
        pred_z = model(torch.tensor(xz, dtype=torch.float32, device=device)).cpu().numpy()
    pred = pred_z * norm["y_scale"] + norm["y_center"]
    pred_mw, pred_pdi, pred_conc = pred[:, 0], pred[:, 1], pred[:, 2]
    score = np.array(
        [property_score_from_values(float(mw), float(pdi), float(conc)) for mw, pdi, conc in zip(pred_mw, pred_pdi, pred_conc)],
        dtype=np.float32,
    )
    return pred_mw, pred_pdi, pred_conc, score

def run_one(seed, initial_rows, all_rows, all_by_reaction, all_by_batch):
    current = {row["Reaction"]: all_by_reaction[row["Reaction"]] for row in initial_rows}
    tested_batches = {row["Batch"] for row in current.values()}
    trajectory = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, MAX_ITERATIONS + 1):
        candidate_rows = [row for row in all_rows if row["Batch"] not in tested_batches]
        if not candidate_rows:
            stop_reason = "candidate_space_exhausted"
            break

        model, norm, train_loss, valid_loss, device = train_model(list(current.values()), seed + iteration)
        pred_mw, pred_pdi, pred_conc, score = predict_rows(model, norm, candidate_rows, device)
        best_idx = int(np.argmin(score))
        recommended = candidate_rows[best_idx]
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
            "Property_score": property_score(recommended),
            "Target_property_distance": target_distance(recommended),
            "Predicted_Mw": float(pred_mw[best_idx]),
            "Predicted_PDI": float(pred_pdi[best_idx]),
            "Predicted_Conc_Mean": float(pred_conc[best_idx]),
            "Recommendation_Score": float(score[best_idx]),
            "Train_Loss": train_loss,
            "Valid_Loss": valid_loss,
            "Batch_best_reaction": best_row["Reaction"],
            "Batch_best_Mw": best_row.get("Mw"),
            "Batch_best_PDI": best_row.get("PDI"),
            "Batch_best_Conc_Mean": best_row.get("Conc_Mean"),
            "Batch_best_target_hit_count": best["Target_hit_count"],
            "Batch_best_property_score": best["Property_score"],
            "Batch_best_property_distance": best["Target_property_distance"],
            "Batch_success_count": len(successes),
            "Success_Reactions": ";".join(item["row"]["Reaction"] for item in successes),
            "Train_Size": len(current),
            "Remaining_Batches": len({row["Batch"] for row in candidate_rows}) - 1,
            "Model": f"ConditionOnlyTransformer(d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NHEAD})",
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
