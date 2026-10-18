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
from torch.utils.data import DataLoader, TensorDataset
from baselines.runtime import load_engine
base = load_engine('descriptor_core', RUNTIME_CONFIG)

def transformed_property_vector(record):
    mw, pdi, conc = base.property_vector(record)
    return np.array([np.log1p(mw), pdi, conc / 100.0], dtype=np.float32)

def make_training_arrays(rows):
    x = np.vstack([base.condition_vector(row) for row in rows]).astype(np.float32)
    y = np.vstack([transformed_property_vector(row) for row in rows]).astype(np.float32)
    return x, y

def matched_property_score_from_values(mw, pdi, conc):
    mw_penalty = max(0.0, base.SUCCESS_MW_LOW - mw) / 500.0
    mw_penalty += max(0.0, mw - base.SUCCESS_MW_HIGH) / 500.0
    pdi_penalty = max(0.0, pdi - base.SUCCESS_PDI_MAX) / 0.10
    conc_penalty = max(0.0, base.SUCCESS_CONC_MIN - conc) / 10.0
    target_pull = (
        abs(mw - base.TARGET_MW) / 1000.0
        + abs(pdi - base.TARGET_PDI) / 0.20
        + abs(conc - base.TARGET_CONC) / 20.0
    )
    return float(
        MW_WINDOW_WEIGHT * mw_penalty
        + PDI_WINDOW_WEIGHT * pdi_penalty
        + CONC_WINDOW_WEIGHT * conc_penalty
        + TARGET_PULL_WEIGHT * target_pull
    )

def matched_property_score(record):
    mw, pdi, conc = base.property_vector(record)
    return matched_property_score_from_values(float(mw), float(pdi), float(conc))

def matched_train_model(train_rows, _round_seed):
    x, y = make_training_arrays(train_rows)
    x_center, x_scale, y_center, y_scale = base.normalize(x, y)
    xz = (x - x_center) / x_scale
    yz = (y - y_center) / y_scale

    valid_mask = np.array(
        [row["Suffix"] in VALID_SUFFIXES for row in train_rows],
        dtype=bool,
    )
    if valid_mask.sum() >= 4 and (~valid_mask).sum() >= 8:
        train_mask = ~valid_mask
    else:
        train_mask = np.ones(len(train_rows), dtype=bool)
        valid_mask = np.zeros(len(train_rows), dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.ConditionOnlyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    train_ds = TensorDataset(
        torch.tensor(xz[train_mask], dtype=torch.float32),
        torch.tensor(yz[train_mask], dtype=torch.float32),
    )
    loader = DataLoader(
        train_ds,
        batch_size=min(base.BATCH_SIZE, len(train_ds)),
        shuffle=True,
    )

    valid_x = torch.tensor(xz[valid_mask], dtype=torch.float32, device=device)
    valid_y = torch.tensor(yz[valid_mask], dtype=torch.float32, device=device)
    best_state = None
    best_valid = float("inf")
    train_loss_value = float("inf")

    for _ in range(base.EPOCHS):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = base.weighted_mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss_value = float(np.mean(losses))

        if valid_mask.any():
            model.eval()
            with torch.no_grad():
                valid_loss = float(base.weighted_mse(model(valid_x), valid_y).detach().cpu())
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_valid = float("nan")

    norm = {
        "x_center": x_center,
        "x_scale": x_scale,
        "y_center": y_center,
        "y_scale": y_scale,
    }
    return model, norm, train_loss_value, best_valid, device

def matched_predict_rows(model, norm, rows, device):
    x = np.vstack([base.condition_vector(row) for row in rows]).astype(np.float32)
    xz = (x - norm["x_center"]) / norm["x_scale"]
    with torch.no_grad():
        pred_z = model(torch.tensor(xz, dtype=torch.float32, device=device)).cpu().numpy()
    pred_transformed = pred_z * norm["y_scale"] + norm["y_center"]
    pred_mw = np.expm1(pred_transformed[:, 0])
    pred_pdi = pred_transformed[:, 1]
    pred_conc = pred_transformed[:, 2] * 100.0
    score = np.array(
        [
            matched_property_score_from_values(float(mw), float(pdi), float(conc))
            for mw, pdi, conc in zip(pred_mw, pred_pdi, pred_conc)
        ],
        dtype=np.float32,
    )
    return pred_mw, pred_pdi, pred_conc, score
