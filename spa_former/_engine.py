"""Spectral state navigation training engine, configured by the public runner.

The raw-patch adapter replaces the compatibility patch projection and input
builders. Their initialization order is retained for numerical parity.
"""

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Install the dependencies in requirements.txt."
    ) from exc


SETTINGS = globals().get("RUNTIME_CONFIG")
if SETTINGS is None:
    raise RuntimeError("Use the spa_former runner to configure paths and a recorded run seed.")
INITIAL_XLSX = Path(SETTINGS["initial_file"])
ALL_XLSX = Path(SETTINGS["all_file"])
SPECTRA_ROOT = Path(SETTINGS["spectra_root"])
OUT_ROOT = Path(SETTINGS["run_directory"])
TARGET = {"Mw": 10000.0, "PDI": 1.45, "Conc_Mean": 80.0}
MW_SUCCESS_LOW, MW_SUCCESS_HIGH = 9000.0, 11000.0
PDI_SUCCESS_MAX, CONC_SUCCESS_MIN = 1.50, 70.0
VALID_SUFFIXES = {1, 5, 10}
PATCH_COUNT = {"GPC": 20, "NMR": 20, "IR": 20}
# Compatibility initialization only. RawPatchClosedLoop uses full intensity sequences.
PATCH_STATS = 5
NMR_LOCAL_NORMALIZATION = False
NMR_PEAK_MAD_FACTOR = 6.0
NMR_NORMALIZATION_EPS = 1e-8
MODALITIES = ["GPC", "NMR", "IR"]
TOKEN_COUNT = sum(PATCH_COUNT[modality] for modality in MODALITIES)
MAX_ROUNDS = int(SETTINGS["max_rounds"])
EPOCHS_PER_ROUND = int(SETTINGS["epochs"])
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
SEED = int(SETTINGS["seed"])
DISTANCE_LOSS_WEIGHT, MODALITY_DISTANCE_LOSS_WEIGHT = 0.8, 0.4
MW_LOSS_WEIGHT, PDI_LOSS_WEIGHT, CONC_LOSS_WEIGHT = 1.8, 2.4, 2.2
ACQ_DISTANCE_WEIGHT = 0.0
ACQ_MW_WINDOW_WEIGHT, ACQ_PDI_WINDOW_WEIGHT, ACQ_CONC_WINDOW_WEIGHT = 1.0, 1.2, 1.2
ACQ_TARGET_PULL_WEIGHT, ACQ_BATCH_TOP_K = 0.60, 1
ACQ_MODE = "property_guided"
ACQ_MODALITY_WEIGHTS = {"GPC": 1.0, "NMR": 1.0, "IR": 1.0}
ACQ_MODALITY_BALANCE_WEIGHT = 0.0
MULTI_DISTANCE_OUTPUT, OUTPUT_DIM = True, 7
DISTANCE_MODALITIES = ["GPC", "NMR", "IR"]
SPECTRA_MODALITIES = list(dict.fromkeys([*MODALITIES, *DISTANCE_MODALITIES]))
FUSION_MODE = "cross_attention"
D_MODEL, NHEAD, NUM_LAYERS, FF_MULT, DROPOUT = 96, 4, 3, 4, 0.10
USE_GATED_FUSION, USE_INFO_NCE = False, False
INFO_NCE_WEIGHT, INFO_NCE_TEMPERATURE = 0.05, 0.20


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reaction_prefix(reaction_id):
    return str(reaction_id).split("-")[0]


def reaction_suffix(reaction_id):
    return int(str(reaction_id).split("-")[1])


def read_workbook(path, reaction_col):
    with Path(path).open("rb") as input_stream:
        wb = openpyxl.load_workbook(input_stream, data_only=True, read_only=True)
        try:
            ws = wb.active
            headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            idx = {h: i for i, h in enumerate(headers)}
            if "Conversion" in idx and "Conc_Mean" not in idx:
                idx["Conc_Mean"] = idx["Conversion"]
            rows = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not any(value is not None and value != "" for value in row):
                    continue
                reaction = str(row[idx[reaction_col]])
                record = {h: row[i] for h, i in idx.items()}
                record["Reaction"] = reaction
                record["Batch"] = reaction_prefix(reaction)
                record["Suffix"] = reaction_suffix(reaction)
                rows.append(record)
            return rows
        finally:
            wb.close()
    

def condition_vector(record):
    return np.array(
        [
            float(record["Total_mols"]),
            float(record["Initiator"]),
            float(record["Time"]),
            float(record["Temp"]),
        ],
        dtype=np.float32,
    )


def metric_vector(record):
    return np.array(
        [
            float(record.get("Mw") or 0.0),
            float(record.get("PDI") or 0.0),
            float(record.get("Conc_Mean") or 0.0),
        ],
        dtype=np.float32,
    )


def is_metric_valid(record):
    metric = metric_vector(record)
    return bool(metric[0] > 0 and metric[1] > 0 and metric[2] > 0)


def is_success(record):
    mw, pdi, conc = metric_vector(record)
    return MW_SUCCESS_LOW <= mw <= MW_SUCCESS_HIGH and pdi < PDI_SUCCESS_MAX and conc > CONC_SUCCESS_MIN


def read_spectrum(modality, reaction):
    path = SPECTRA_ROOT / modality / f"{reaction}.csv"
    if not path.exists():
        return None
    if modality == "NMR":
        arr = np.loadtxt(path, dtype=np.float32)
    else:
        arr = np.loadtxt(path, dtype=np.float32, delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]


def interp_to_grid(x, y, target_x):
    if x[0] > x[-1]:
        xx = x[::-1]
        yy = y[::-1]
    else:
        xx = x
        yy = y
    if target_x[0] > target_x[-1]:
        tx = target_x[::-1]
        reverse = True
    else:
        tx = target_x
        reverse = False
    out = np.interp(tx, xx, yy, left=yy[0], right=yy[-1]).astype(np.float32)
    return out[::-1] if reverse else out


def knn_weights(target, matrix, k=24):
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    z = (matrix - center) / scale
    t = (target - center) / scale
    distances = np.linalg.norm(z - t, axis=1)
    order = np.argsort(distances)[: min(k, len(distances))]
    local = distances[order]
    bandwidth = np.median(local[local > 0]) if np.any(local > 0) else 1.0
    bandwidth = max(float(bandwidth), 1e-6)
    weights = np.exp(-0.5 * (local / bandwidth) ** 2)
    weights = weights / weights.sum()
    return order, weights


def robust_mad(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad


def normalize_nmr_patch_if_peaked(patch):
    if not NMR_LOCAL_NORMALIZATION:
        return patch
    p5 = float(np.percentile(patch, 5))
    p99 = float(np.percentile(patch, 99))
    dynamic_range = p99 - p5
    noise = float(robust_mad(patch))
    if noise <= NMR_NORMALIZATION_EPS:
        noise = float(np.std(patch))
    has_peak = dynamic_range > NMR_PEAK_MAD_FACTOR * max(noise, NMR_NORMALIZATION_EPS)
    if not has_peak:
        return patch
    return (patch - p5) / max(dynamic_range, NMR_NORMALIZATION_EPS)


def patch_stats(y, patch_count, modality=None):
    patches = np.array_split(y.astype(np.float64), patch_count)
    rows = []
    for patch in patches:
        if modality == "NMR":
            patch = normalize_nmr_patch_if_peaked(patch)
        rows.append(
            [
                float(patch.mean()),
                float(patch.std()),
                float(patch.min()),
                float(patch.max()),
                float(np.sqrt(np.mean(patch * patch))),
            ]
        )
    return np.array(rows, dtype=np.float32)


def standardize_fit(x):
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def standardize_apply(x, center, scale):
    return ((x - center) / scale).astype(np.float32)


class SpectraConditionTransformer(nn.Module):
    def __init__(
        self,
        token_dim=5,
        condition_dim=4,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
    ):
        super().__init__()
        self.patch_proj = nn.Linear(token_dim, d_model)
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.modality_embed = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * FF_MULT,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model, OUTPUT_DIM),
        )
        modality_ids = []
        for idx, modality in enumerate(MODALITIES, start=1):
            modality_ids.extend([idx] * PATCH_COUNT[modality])
        self.register_buffer("modality_ids", torch.tensor(modality_ids, dtype=torch.long), persistent=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + TOKEN_COUNT, d_model))
        self.use_cross_attention = FUSION_MODE == "cross_attention" and len(MODALITIES) > 1
        if self.use_cross_attention:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(d_model, nhead, dropout=0.10, batch_first=True)
                    for _ in MODALITIES
                ]
            )
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in MODALITIES])
            self.cross_ffn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model * 2),
                        nn.GELU(),
                        nn.Dropout(DROPOUT),
                        nn.Linear(d_model * 2, d_model),
                    )
                    for _ in MODALITIES
                ]
            )
        self.use_gated_fusion = USE_GATED_FUSION and len(MODALITIES) > 1
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * (len(MODALITIES) + 1)),
            nn.Linear(d_model * (len(MODALITIES) + 1), d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model, len(MODALITIES)),
        ) if self.use_gated_fusion else None

    def cross_modal_fusion(self, patch_token):
        if not self.use_cross_attention:
            return patch_token
        chunks = []
        start = 0
        for modality in MODALITIES:
            end = start + PATCH_COUNT[modality]
            chunks.append(patch_token[:, start:end, :])
            start = end

        fused = []
        for idx, query in enumerate(chunks):
            key_value = torch.cat([chunk for j, chunk in enumerate(chunks) if j != idx], dim=1)
            attended, _ = self.cross_attn[idx](query, key_value, key_value, need_weights=False)
            updated = self.cross_norm[idx](query + attended)
            updated = updated + self.cross_ffn[idx](updated)
            fused.append(updated)
        return torch.cat(fused, dim=1)

    def modality_chunks(self, patch_token):
        chunks = []
        start = 0
        for modality in MODALITIES:
            end = start + PATCH_COUNT[modality]
            chunks.append(patch_token[:, start:end, :])
            start = end
        return chunks

    def forward_features(self, patch_tokens, conditions):
        # patch_tokens: [B, TOKEN_COUNT, 5], conditions: [B, 4]
        cond_token = self.condition_proj(conditions).unsqueeze(1)
        patch_token = self.patch_proj(patch_tokens)
        patch_token = patch_token + self.modality_embed(self.modality_ids.to(patch_tokens.device)).unsqueeze(0)
        patch_token = self.cross_modal_fusion(patch_token)
        x = torch.cat([cond_token + self.modality_embed.weight[0].view(1, 1, -1), patch_token], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        h = self.encoder(x)
        patch_h = h[:, 1:, :]
        modality_reprs = torch.stack([chunk.mean(dim=1) for chunk in self.modality_chunks(patch_h)], dim=1)
        if self.use_gated_fusion:
            gate_input = torch.cat([h[:, 0, :], modality_reprs.reshape(modality_reprs.shape[0], -1)], dim=1)
            gate_weights = torch.softmax(self.gate(gate_input), dim=1)
            fused = h[:, 0, :] + torch.sum(gate_weights.unsqueeze(-1) * modality_reprs, dim=1)
        else:
            gate_weights = torch.full(
                (patch_tokens.shape[0], len(MODALITIES)),
                1.0 / max(1, len(MODALITIES)),
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            fused = h[:, 0, :]
        return fused, modality_reprs, gate_weights

    def forward(self, patch_tokens, conditions, return_aux=False):
        fused, modality_reprs, gate_weights = self.forward_features(patch_tokens, conditions)
        pred = self.head(fused)
        if return_aux:
            return pred, {"modality_reprs": modality_reprs, "gate_weights": gate_weights}
        return pred


@dataclass
class RoundResult:
    iteration: int
    recommended_reaction: str
    recommended_batch: str
    train_loss: float
    valid_loss: float
    batch_success_count: int
    predicted_distance: float
    predicted_mw: float
    predicted_pdi: float
    predicted_conc_mean: float
    nearest_success_reaction: str


class TransformerClosedLoop:
    def __init__(self):
        set_seed()
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_rows = read_workbook(INITIAL_XLSX, "Reaction")
        self.all_rows = read_workbook(ALL_XLSX, "Reaction_ID")
        self.all_by_reaction = {row["Reaction"]: row for row in self.all_rows}
        self.all_by_batch = {}
        for row in self.all_rows:
            self.all_by_batch.setdefault(row["Batch"], []).append(row)
        for batch_rows in self.all_by_batch.values():
            batch_rows.sort(key=lambda r: r["Suffix"])

        self.current = {
            row["Reaction"]: self.all_by_reaction[row["Reaction"]]
            for row in self.initial_rows
        }
        self.spectra = {}
        self.target_spectra = {}
        self.load_initial_spectra()
        self.predict_target_spectra()

    def load_initial_spectra(self):
        for reaction in sorted(self.current):
            self.spectra[reaction] = {}
            for modality in SPECTRA_MODALITIES:
                loaded = read_spectrum(modality, reaction)
                if loaded is not None:
                    self.spectra[reaction][modality] = loaded

    def synthesize_spectrum(self, modality, target_vector, use_metrics=True):
        reactions = []
        features = []
        for reaction, record in self.current.items():
            if modality not in self.spectra.get(reaction, {}):
                continue
            if use_metrics and not is_metric_valid(record):
                continue
            reactions.append(reaction)
            features.append(metric_vector(record) if use_metrics else condition_vector(record))
        if not reactions:
            raise RuntimeError(f"No available spectra for {modality}")
        matrix = np.vstack(features)
        order, weights = knn_weights(target_vector.astype(np.float32), matrix, k=24)
        base_reaction = reactions[int(order[0])]
        target_x = self.spectra[base_reaction][modality][0]
        y = np.zeros_like(target_x, dtype=np.float32)
        for idx, weight in zip(order, weights):
            reaction = reactions[int(idx)]
            x_i, y_i = self.spectra[reaction][modality]
            y += float(weight) * interp_to_grid(x_i, y_i, target_x)
        return target_x, y

    def predict_target_spectra(self):
        target = np.array([TARGET["Mw"], TARGET["PDI"], TARGET["Conc_Mean"]], dtype=np.float32)
        target_dir = OUT_ROOT / "target_spectra"
        for modality in DISTANCE_MODALITIES:
            x, y = self.synthesize_spectrum(modality, target, use_metrics=True)
            self.target_spectra[modality] = (x, y)
            self.write_spectrum(target_dir / modality / "target_Mw10000_PDI1p45_Conc80.csv", modality, x, y)

    def write_spectrum(self, path, modality, x, y):
        path.parent.mkdir(parents=True, exist_ok=True)
        delimiter = "\t" if modality == "NMR" else ","
        with path.open("w", newline="", encoding="utf-8") as handle:
            if modality != "NMR":
                handle.write("x,y\n")
            writer = csv.writer(handle, delimiter=delimiter)
            for xv, yv in zip(x, y):
                writer.writerow([f"{float(xv):.12g}", f"{float(yv):.12g}"])

    def ensure_spectra(self, record):
        reaction = record["Reaction"]
        self.spectra.setdefault(reaction, {})
        for modality in SPECTRA_MODALITIES:
            if modality in self.spectra[reaction]:
                continue
            loaded = read_spectrum(modality, reaction)
            if loaded is not None:
                self.spectra[reaction][modality] = loaded
                continue
            if is_metric_valid(record):
                x, y = self.synthesize_spectrum(modality, metric_vector(record), use_metrics=True)
            else:
                x, y = self.synthesize_spectrum(modality, condition_vector(record), use_metrics=False)
            self.spectra[reaction][modality] = (x, y)

    def spectral_distances(self, reaction):
        distances = {}
        for modality in DISTANCE_MODALITIES:
            target_x, target_y = self.target_spectra[modality]
            x, y = self.spectra[reaction][modality]
            yy = interp_to_grid(x, y, target_x)
            distances[modality] = float(np.linalg.norm(yy.astype(np.float64) - target_y.astype(np.float64)))
        return distances

    def spectral_distance(self, reaction):
        return sum(self.spectral_distances(reaction).values())
        return total

    def build_arrays(self):
        reactions = sorted(self.current)
        cond_rows = []
        patch_rows = []
        target_rows = []
        valid = []
        metric_valid = []
        records = []
        for reaction in reactions:
            record = self.current[reaction]
            self.ensure_spectra(record)
            tokens = []
            for modality in MODALITIES:
                _, y = self.spectra[reaction][modality]
                tokens.append(patch_stats(y, PATCH_COUNT[modality], modality))
            distances = self.spectral_distances(reaction)
            distance = sum(distances.values())
            metric = metric_vector(record)
            cond_rows.append(condition_vector(record))
            patch_rows.append(np.vstack(tokens))
            if MULTI_DISTANCE_OUTPUT:
                target_rows.append(
                    [
                        math.log1p(distances.get("GPC", 0.0)),
                        math.log1p(distances.get("NMR", 0.0)),
                        math.log1p(distances.get("IR", 0.0)),
                        math.log1p(distance),
                        math.log1p(metric[0]),
                        metric[1],
                        metric[2] / 100.0,
                    ]
                )
            else:
                target_rows.append(
                    [
                        math.log1p(distance),
                        math.log1p(metric[0]),
                        metric[1],
                        metric[2] / 100.0,
                    ]
                )
            valid.append(record["Suffix"] in VALID_SUFFIXES)
            metric_valid.append(is_metric_valid(record))
            records.append(record)
        return (
            reactions,
            records,
            np.stack(cond_rows).astype(np.float32),
            np.stack(patch_rows).astype(np.float32),
            np.array(target_rows, dtype=np.float32),
            np.array(valid, dtype=bool),
            np.array(metric_valid, dtype=bool),
        )

    def normalize_dataset(self):
        reactions, records, cond, patch, target, valid, metric_valid = self.build_arrays()
        train = ~valid
        c_center, c_scale = standardize_fit(cond[train])
        p_center, p_scale = standardize_fit(patch.reshape(len(patch), -1)[train])
        y_center = np.zeros(target.shape[1], dtype=np.float32)
        y_scale = np.ones(target.shape[1], dtype=np.float32)
        distance_cols = 4 if MULTI_DISTANCE_OUTPUT else 1
        y_center[:distance_cols], y_scale[:distance_cols] = standardize_fit(target[train, :distance_cols])
        metric_train = train & metric_valid
        if metric_train.any():
            y_center[distance_cols:], y_scale[distance_cols:] = standardize_fit(target[metric_train, distance_cols:])

        cond_z = standardize_apply(cond, c_center, c_scale)
        patch_flat_z = standardize_apply(patch.reshape(len(patch), -1), p_center, p_scale)
        patch_z = patch_flat_z.reshape(patch.shape)
        y_z = standardize_apply(target, y_center, y_scale)
        norm = {
            "condition_center": c_center,
            "condition_scale": c_scale,
            "patch_center": p_center,
            "patch_scale": p_scale,
            "target_center": y_center,
            "target_scale": y_scale,
        }
        return reactions, records, cond_z, patch_z, y_z, target, valid, metric_valid, norm

    @staticmethod
    def multitask_loss(pred, target, metric_mask):
        if MULTI_DISTANCE_OUTPUT:
            modality_distance_loss = torch.mean((pred[:, :3] - target[:, :3]) ** 2)
            total_distance_loss = torch.mean((pred[:, 3] - target[:, 3]) ** 2)
            distance_loss = (
                MODALITY_DISTANCE_LOSS_WEIGHT * modality_distance_loss
                + DISTANCE_LOSS_WEIGHT * total_distance_loss
            )
            metric_start = 4
        else:
            distance_loss = DISTANCE_LOSS_WEIGHT * torch.mean((pred[:, 0] - target[:, 0]) ** 2)
            metric_start = 1
        if torch.any(metric_mask):
            metric_errors = (pred[metric_mask, metric_start:] - target[metric_mask, metric_start:]) ** 2
            metric_loss = (
                MW_LOSS_WEIGHT * torch.mean(metric_errors[:, 0])
                + PDI_LOSS_WEIGHT * torch.mean(metric_errors[:, 1])
                + CONC_LOSS_WEIGHT * torch.mean(metric_errors[:, 2])
            )
            return distance_loss + metric_loss
        return distance_loss

    @staticmethod
    def info_nce_loss(modality_reprs):
        if not (USE_INFO_NCE and modality_reprs.shape[1] > 1 and modality_reprs.shape[0] > 1):
            return modality_reprs.new_tensor(0.0)
        z = torch.nn.functional.normalize(modality_reprs, dim=-1)
        labels = torch.arange(z.shape[0], device=z.device)
        losses = []
        for i in range(z.shape[1]):
            for j in range(i + 1, z.shape[1]):
                logits = torch.matmul(z[:, i, :], z[:, j, :].T) / INFO_NCE_TEMPERATURE
                losses.append(torch.nn.functional.cross_entropy(logits, labels))
                losses.append(torch.nn.functional.cross_entropy(logits.T, labels))
        return torch.stack(losses).mean()

    def train_one_round(self, iteration):
        reactions, records, cond, patch, y_z, y_raw, valid, metric_valid, norm = self.normalize_dataset()
        train = ~valid
        model = SpectraConditionTransformer().to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_ds = TensorDataset(
            torch.from_numpy(patch[train]),
            torch.from_numpy(cond[train]),
            torch.from_numpy(y_z[train]),
            torch.from_numpy(metric_valid[train]),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_patch = torch.from_numpy(patch[valid]).to(self.device)
        valid_cond = torch.from_numpy(cond[valid]).to(self.device)
        valid_y = torch.from_numpy(y_z[valid]).to(self.device)
        valid_metric_mask = torch.from_numpy(metric_valid[valid]).to(self.device)

        best_state = None
        best_valid = float("inf")
        train_loss_value = float("inf")
        for _ in range(EPOCHS_PER_ROUND):
            model.train()
            losses = []
            for xb_patch, xb_cond, yb, mb in train_loader:
                xb_patch = xb_patch.to(self.device)
                xb_cond = xb_cond.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)
                pred, aux = model(xb_patch, xb_cond, return_aux=True)
                loss = self.multitask_loss(pred, yb, mb)
                if USE_INFO_NCE and len(MODALITIES) > 1:
                    loss = loss + INFO_NCE_WEIGHT * self.info_nce_loss(aux["modality_reprs"])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            train_loss_value = float(np.mean(losses))
            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_patch, valid_cond)
                valid_loss = float(
                    self.multitask_loss(valid_pred, valid_y, valid_metric_mask).detach().cpu()
                )
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model_dir = OUT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "normalization": {k: v.tolist() for k, v in norm.items()},
                "target": TARGET,
                "iteration": iteration,
                "multi_distance_output": MULTI_DISTANCE_OUTPUT,
                "output_dim": OUTPUT_DIM,
            },
            model_dir / f"round_{iteration:03d}.pt",
        )
        return model, norm, train_loss_value, best_valid

    @staticmethod
    def count_parameters(model):
        return int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    def candidate_tokens(self, candidate_rows, norm):
        cond = np.vstack([condition_vector(row) for row in candidate_rows]).astype(np.float32)
        cond_z = standardize_apply(cond, norm["condition_center"], norm["condition_scale"])

        # Candidate spectra are unknown before experiment. Use condition-nearest current spectra
        # as virtual patch tokens, so the Transformer still receives spectra-shaped input.
        current_reactions = sorted(self.current)
        current_cond = np.vstack([condition_vector(self.current[r]) for r in current_reactions])
        current_patch = []
        for reaction in current_reactions:
            tokens = []
            for modality in MODALITIES:
                _, y = self.spectra[reaction][modality]
                tokens.append(patch_stats(y, PATCH_COUNT[modality], modality))
            current_patch.append(np.vstack(tokens))
        current_patch = np.stack(current_patch)
        current_center, current_scale = standardize_fit(current_cond)
        current_z = standardize_apply(current_cond, current_center, current_scale)
        candidate_z_for_knn = standardize_apply(cond, current_center, current_scale)
        patch_pred = []
        for z in candidate_z_for_knn:
            distances = np.linalg.norm(current_z - z, axis=1)
            order = np.argsort(distances)[:8]
            weights = np.exp(-distances[order] / max(float(np.median(distances[order])), 1e-6))
            weights = weights / weights.sum()
            patch_pred.append(np.tensordot(weights, current_patch[order], axes=(0, 0)))
        patch_pred = np.stack(patch_pred).astype(np.float32)
        flat = patch_pred.reshape(len(patch_pred), -1)
        flat_z = standardize_apply(flat, norm["patch_center"], norm["patch_scale"])
        patch_z = flat_z.reshape(patch_pred.shape)
        return cond_z, patch_z

    def recommend_next_batch(self, model, norm):
        tested = {reaction_prefix(r) for r in self.current}
        candidate_rows = [row for row in self.all_rows if row["Batch"] not in tested]
        cond_z, patch_z = self.candidate_tokens(candidate_rows, norm)
        model.eval()
        preds = []
        gates = []
        with torch.no_grad():
            for start in range(0, len(candidate_rows), 512):
                p = torch.from_numpy(patch_z[start : start + 512]).to(self.device)
                c = torch.from_numpy(cond_z[start : start + 512]).to(self.device)
                out, aux = model(p, c, return_aux=True)
                gates.append(aux["gate_weights"].detach().cpu().numpy())
                out = out.detach().cpu().numpy()
                preds.append(out)
        pred_z = np.vstack(preds)
        gate_array = np.vstack(gates) if gates else np.zeros((len(candidate_rows), len(MODALITIES)), dtype=np.float32)
        pred_raw = pred_z * norm["target_scale"] + norm["target_center"]
        if MULTI_DISTANCE_OUTPUT:
            pred_gpc_distance = np.expm1(pred_raw[:, 0])
            pred_nmr_distance = np.expm1(pred_raw[:, 1])
            pred_ir_distance = np.expm1(pred_raw[:, 2])
            pred_distance = np.expm1(pred_raw[:, 3])
            metric_start = 4
            modality_distance_raw = {
                "GPC": pred_raw[:, 0],
                "NMR": pred_raw[:, 1],
                "IR": pred_raw[:, 2],
            }
            if ACQ_MODE == "modality_distance_only":
                distance_score_raw = np.sum([modality_distance_raw[m] for m in MODALITIES], axis=0)
                pred_distance = np.sum(
                    [
                        {"GPC": pred_gpc_distance, "NMR": pred_nmr_distance, "IR": pred_ir_distance}[m]
                        for m in MODALITIES
                    ],
                    axis=0,
                )
            elif ACQ_MODE == "weighted_modality_property":
                weighted_terms = [
                    ACQ_MODALITY_WEIGHTS[m] * modality_distance_raw[m]
                    for m in MODALITIES
                ]
                distance_score_raw = np.sum(weighted_terms, axis=0) / max(
                    sum(ACQ_MODALITY_WEIGHTS[m] for m in MODALITIES), 1e-9
                )
                pred_distance = np.sum(
                    [
                        ACQ_MODALITY_WEIGHTS[m]
                        * {"GPC": pred_gpc_distance, "NMR": pred_nmr_distance, "IR": pred_ir_distance}[m]
                        for m in MODALITIES
                    ],
                    axis=0,
                ) / max(sum(ACQ_MODALITY_WEIGHTS[m] for m in MODALITIES), 1e-9)
            else:
                distance_score_raw = pred_raw[:, 3]
        else:
            pred_gpc_distance = np.full(len(pred_raw), np.nan)
            pred_nmr_distance = np.full(len(pred_raw), np.nan)
            pred_ir_distance = np.full(len(pred_raw), np.nan)
            pred_distance = np.expm1(pred_raw[:, 0])
            metric_start = 1
            distance_score_raw = pred_raw[:, 0]
        pred_mw = np.expm1(pred_raw[:, metric_start])
        pred_pdi = pred_raw[:, metric_start + 1]
        pred_conc = pred_raw[:, metric_start + 2] * 100.0

        distance_score = (distance_score_raw - distance_score_raw.mean()) / max(float(distance_score_raw.std()), 1e-9)
        mw_penalty = np.maximum(0, MW_SUCCESS_LOW - pred_mw) / 500.0
        mw_penalty += np.maximum(0, pred_mw - MW_SUCCESS_HIGH) / 500.0
        pdi_penalty = np.maximum(0, pred_pdi - PDI_SUCCESS_MAX) / 0.10
        conc_penalty = np.maximum(0, CONC_SUCCESS_MIN - pred_conc) / 10.0
        target_pull = (
            np.abs(pred_mw - TARGET["Mw"]) / 1000.0
            + np.abs(pred_pdi - TARGET["PDI"]) / 0.20
            + np.abs(pred_conc - TARGET["Conc_Mean"]) / 20.0
        )
        # Score a batch by its lowest-scoring time-resolved member.
        if ACQ_MODE == "modality_distance_only":
            acquisition = distance_score
        elif ACQ_MODE == "weighted_modality_property":
            acquisition = (
                ACQ_DISTANCE_WEIGHT * distance_score
                + ACQ_MW_WINDOW_WEIGHT * mw_penalty
                + ACQ_PDI_WINDOW_WEIGHT * pdi_penalty
                + ACQ_CONC_WINDOW_WEIGHT * conc_penalty
                + ACQ_TARGET_PULL_WEIGHT * target_pull
            )
            if MULTI_DISTANCE_OUTPUT and len(MODALITIES) > 1 and ACQ_MODALITY_BALANCE_WEIGHT > 0:
                selected = np.vstack([modality_distance_raw[m] for m in MODALITIES]).T
                selected_z = (selected - selected.mean(axis=0, keepdims=True)) / np.maximum(
                    selected.std(axis=0, keepdims=True), 1e-9
                )
                balance_penalty = selected_z.max(axis=1) - selected_z.min(axis=1)
                acquisition = acquisition + ACQ_MODALITY_BALANCE_WEIGHT * balance_penalty
        else:
            acquisition = (
                ACQ_DISTANCE_WEIGHT * distance_score
                + ACQ_MW_WINDOW_WEIGHT * mw_penalty
                + ACQ_PDI_WINDOW_WEIGHT * pdi_penalty
                + ACQ_CONC_WINDOW_WEIGHT * conc_penalty
                + ACQ_TARGET_PULL_WEIGHT * target_pull
            )
        batch_items = {}
        for idx, row in enumerate(candidate_rows):
            batch_items.setdefault(row["Batch"], []).append((float(acquisition[idx]), idx))
        batch_best = {}
        for batch, items in batch_items.items():
            top_items = sorted(items, key=lambda item: item[0])[: max(1, ACQ_BATCH_TOP_K)]
            batch_score = float(np.mean([item[0] for item in top_items]))
            best_member_idx = int(top_items[0][1])
            batch_best[batch] = (batch_score, best_member_idx)
        best_idx = int(min(batch_best.values(), key=lambda item: item[0])[1])
        best_row = candidate_rows[best_idx]
        batch_rows = self.all_by_batch[best_row["Batch"]]
        success_rows = [row for row in batch_rows if is_success(row)]
        gate_values = {
            f"gate_{modality.lower()}": float(gate_array[best_idx, idx])
            for idx, modality in enumerate(MODALITIES)
        }
        return {
            "best_row": best_row,
            "batch_rows": batch_rows,
            "success_rows": success_rows,
            "predicted_distance": float(pred_distance[best_idx]),
            "predicted_gpc_distance": float(pred_gpc_distance[best_idx]),
            "predicted_nmr_distance": float(pred_nmr_distance[best_idx]),
            "predicted_ir_distance": float(pred_ir_distance[best_idx]),
            "predicted_mw": float(pred_mw[best_idx]),
            "predicted_pdi": float(pred_pdi[best_idx]),
            "predicted_conc": float(pred_conc[best_idx]),
            "acquisition": float(acquisition[best_idx]),
            **gate_values,
        }

    def add_batch(self, batch_rows):
        for row in batch_rows:
            self.current[row["Reaction"]] = row
            self.ensure_spectra(row)

    def run(self):
        logs = []
        stop_reason = "running"
        for iteration in range(1, MAX_ROUNDS + 1):
            model, norm, train_loss, valid_loss = self.train_one_round(iteration)
            rec = self.recommend_next_batch(model, norm)
            best = rec["best_row"]
            success_rows = rec["success_rows"]
            log_row = {
                "Iteration": iteration,
                "Recommended_Reaction": best["Reaction"],
                "Recommended_Batch": best["Batch"],
                "Predicted_Distance": rec["predicted_distance"],
                "Predicted_Mw": rec["predicted_mw"],
                "Predicted_PDI": rec["predicted_pdi"],
                "Predicted_Conc_Mean": rec["predicted_conc"],
                "Acquisition": rec["acquisition"],
                "Train_Loss": train_loss,
                "Valid_Loss": valid_loss,
                "Parameter_Count": self.count_parameters(model),
                "Train_Size": len([r for r in self.current.values() if r["Suffix"] not in VALID_SUFFIXES]),
                "Valid_Size": len([r for r in self.current.values() if r["Suffix"] in VALID_SUFFIXES]),
                "Batch_Success_Count": len(success_rows),
                "Success_Reactions": ";".join(row["Reaction"] for row in success_rows),
            }
            for modality in MODALITIES:
                log_row[f"Gate_{modality}"] = rec.get(f"gate_{modality.lower()}", float("nan"))
            logs.append(log_row)
            self.write_logs(logs, stop_reason)
            print(
                f"ITER {iteration} batch={best['Batch']} reaction={best['Reaction']} "
                f"acq={rec['acquisition']:.6g} success_count={len(success_rows)} "
                f"train_loss={train_loss:.6g} valid_loss={valid_loss:.6g}",
                flush=True,
            )
            if success_rows:
                stop_reason = "success_condition_met"
                self.write_logs(logs, stop_reason)
                break
            self.add_batch(rec["batch_rows"])
        else:
            stop_reason = "max_rounds"
            self.write_logs(logs, stop_reason)
        return logs, stop_reason

    def write_logs(self, logs, stop_reason):
        if logs:
            with (OUT_ROOT / "iteration_log.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
                writer.writeheader()
                writer.writerows(logs)
        summary = {
            "model": "SpectraConditionTransformer",
            "stop_reason": stop_reason,
            "rounds": len(logs),
            "target": TARGET,
            "success_criteria": {
                "Mw": [MW_SUCCESS_LOW, MW_SUCCESS_HIGH],
                "PDI": f"<{PDI_SUCCESS_MAX}",
                "Conc_Mean": f">{CONC_SUCCESS_MIN}",
            },
            "patching": {
                modality: f"{PATCH_COUNT[modality]} patches x {PATCH_STATS} statistics"
                for modality in MODALITIES
            },
            "nmr_patch_preprocessing": {
                "local_normalization": NMR_LOCAL_NORMALIZATION,
                "peak_detection": "p99-p5 > factor * robust_MAD",
                "peak_mad_factor": NMR_PEAK_MAD_FACTOR,
                "normalization": "(patch - percentile_5) / (percentile_99 - percentile_5 + eps)",
                "applied_only_to_peaked_patches": True,
            },
            "modalities": MODALITIES,
            "distance_modalities": DISTANCE_MODALITIES,
            "token_count": TOKEN_COUNT,
            "device": str(self.device),
            "seed": SEED,
            "initial_xlsx": str(INITIAL_XLSX),
            "spectra_root": str(SPECTRA_ROOT),
            "initial_size": len(self.initial_rows),
            "initial_batches": sorted({row["Batch"] for row in self.initial_rows}),
            "architecture": {
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "ff_mult": FF_MULT,
                "dropout": DROPOUT,
                "parameter_count": logs[-1]["Parameter_Count"] if logs else None,
            },
            "optimization": {
                "multi_distance_output": MULTI_DISTANCE_OUTPUT,
                "output_dim": OUTPUT_DIM,
                "fusion_mode": FUSION_MODE,
                "gated_fusion": USE_GATED_FUSION,
                "info_nce": USE_INFO_NCE,
                "info_nce_weight": INFO_NCE_WEIGHT,
                "info_nce_temperature": INFO_NCE_TEMPERATURE,
                "loss_weights": {
                    "distance": DISTANCE_LOSS_WEIGHT,
                    "modality_distance": MODALITY_DISTANCE_LOSS_WEIGHT,
                    "Mw": MW_LOSS_WEIGHT,
                    "PDI": PDI_LOSS_WEIGHT,
                    "Conc_Mean": CONC_LOSS_WEIGHT,
                },
                "acquisition_weights": {
                    "distance": ACQ_DISTANCE_WEIGHT,
                    "Mw_window": ACQ_MW_WINDOW_WEIGHT,
                    "PDI_window": ACQ_PDI_WINDOW_WEIGHT,
                    "Conc_window": ACQ_CONC_WINDOW_WEIGHT,
                    "target_pull": ACQ_TARGET_PULL_WEIGHT,
                    "batch_top_k": ACQ_BATCH_TOP_K,
                    "mode": ACQ_MODE,
                    "GPC_distance": ACQ_MODALITY_WEIGHTS["GPC"],
                    "NMR_distance": ACQ_MODALITY_WEIGHTS["NMR"],
                    "IR_distance": ACQ_MODALITY_WEIGHTS["IR"],
                    "modality_balance": ACQ_MODALITY_BALANCE_WEIGHT,
                },
            },
            "final": logs[-1] if logs else None,
        }
        with (OUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


