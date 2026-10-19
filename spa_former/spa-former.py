"""Spectral state navigation with full-intensity CNN spectral patches."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from spa_former.cli import main

    main()
    raise SystemExit(0)

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_engine.py"
VERSION = "spectral state navigation"
DIRECTORIES = {"GPC": "GPC", "NMR": "NMR", "IR": "FTIR"}
OUTPUT_NAMES = ["D_GPC", "D_NMR", "D_FTIR", "D_aux", "Mw", "PDI", "Conversion"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def configure(seed, args):
    from .seeds import validate_seeds

    seed = validate_seeds([seed])[0]
    settings = {
        "initial_file": str(Path(args.initial_file).resolve()),
        "all_file": str(Path(args.all_file).resolve()),
        "spectra_root": str(Path(args.spectra_root).resolve()),
        "run_directory": str(Path(args.output).resolve() / "run_seed_{}".format(seed)),
        "seed": seed, "max_rounds": args.max_rounds, "epochs": args.epochs,
        "execution_kind": "smoke" if getattr(args, "smoke", False) else "replay",
    }
    name = "spa_former_runtime_{}".format(seed)
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    base = importlib.util.module_from_spec(spec)
    base.RUNTIME_CONFIG = settings
    sys.modules[name] = base
    spec.loader.exec_module(base)

    def read_spectrum(modality, reaction):
        path = Path(args.spectra_root) / DIRECTORIES[modality] / (reaction + ".csv")
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs = {} if modality == "NMR" else {"delimiter": ",", "skiprows": 1}
        array = np.loadtxt(path, dtype=np.float32, **kwargs)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError("Invalid spectrum: " + str(path))
        return array[:, 0], array[:, 1]

    base.read_spectrum = read_spectrum
    return base, settings


class RawPatchLayout:
    def __init__(self, grids, patch_counts, reference_reaction):
        self.grids = {m: np.asarray(x, dtype=np.float32).copy() for m, x in grids.items()}
        self.modalities = list(grids)
        self.reference_reaction = reference_reaction
        self.entries = {}
        offset = 0
        for modality in self.modalities:
            x = self.grids[modality]
            dx = np.diff(x.astype(np.float64))
            count = patch_counts[modality]
            if len(x) < count or not (np.all(dx > 0) or np.all(dx < 0)):
                raise ValueError("Invalid reference spectral axis: " + modality)
            lengths = [len(part) for part in np.array_split(x, count)]
            width = max(lengths)
            end = offset + count * width
            self.entries[modality] = {
                "points": len(x), "patches": count, "points_per_patch": lengths,
                "patch_width": width, "flat_start": offset, "flat_end": end,
                "padding_points": count * width - len(x),
                "axis_tolerance": max(float(np.median(np.abs(dx))) * 0.02, 1e-6),
                "axis_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "axis_first": float(x[0]), "axis_last": float(x[-1]),
            }
            offset = end
        self.flat_dim = offset

    def validate(self, modality, x, y):
        entry = self.entries[modality]
        if x.shape != self.grids[modality].shape or y.shape != x.shape:
            raise ValueError("Raw spectrum point-count mismatch: " + modality)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite raw spectrum: " + modality)
        if not np.allclose(x, self.grids[modality], rtol=0, atol=entry["axis_tolerance"]):
            raise ValueError("Incompatible spectral grid; refusing silent interpolation: " + modality)

    def encode_modality(self, modality, x, y):
        self.validate(modality, x, y)
        entry = self.entries[modality]
        result = np.zeros((entry["patches"], entry["patch_width"]), dtype=np.float32)
        for index, part in enumerate(np.array_split(y, entry["patches"])):
            result[index, :len(part)] = part
        return result

    def encode(self, spectra):
        return np.concatenate([
            self.encode_modality(modality, *spectra[modality]).ravel()
            for modality in self.modalities
        ])

    def metadata(self):
        return {"reference_reaction": self.reference_reaction, "modalities": self.entries,
                "flat_input_dim": self.flat_dim, "token_count": sum(e["patches"] for e in self.entries.values()),
                "resampling": False, "intensity_statistics": False,
                "padding": "zeros at the end of short patches only",
                "axis_tolerance": "2% of reference sampling interval, to tolerate coordinate rounding only"}


def fit_layout(base):
    initial = base.read_workbook(base.INITIAL_XLSX, "Reaction")
    reaction = min(row["Reaction"] for row in initial if row["Suffix"] not in base.VALID_SUFFIXES)
    grids = {m: base.read_spectrum(m, reaction)[0] for m in base.MODALITIES}
    return RawPatchLayout(grids, base.PATCH_COUNT, reaction)


class PatchSpectralEncoder(nn.Module):
    def __init__(self, patch_width, d_model):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )
        encoded_width = (patch_width + 3) // 4
        encoded_width = (encoded_width + 3) // 4
        self.projection = nn.Linear(32 * encoded_width, d_model)

    def forward(self, patch):
        return self.projection(self.features(patch.unsqueeze(1)).flatten(1))


class ModalityRawProjection(nn.Module):
    def __init__(self, layout, d_model):
        super().__init__()
        self.layout = layout
        self.encoders = nn.ModuleDict({
            m: PatchSpectralEncoder(layout.entries[m]["patch_width"], d_model) for m in layout.modalities
        })

    def forward(self, packed):
        if packed.ndim != 2 or packed.shape[1] != self.layout.flat_dim:
            raise ValueError("Expected a batch of packed complete raw patches.")
        tokens = []
        for modality in self.layout.modalities:
            entry = self.layout.entries[modality]
            patch = packed[:, entry["flat_start"]:entry["flat_end"]].reshape(
                packed.shape[0], entry["patches"], entry["patch_width"])
            embedded = self.encoders[modality](patch.reshape(-1, entry["patch_width"]))
            tokens.append(embedded.reshape(packed.shape[0], entry["patches"], -1))
        return torch.cat(tokens, dim=1)


def install_raw_model(base, layout):
    original = base.SpectraConditionTransformer

    class RawPatchTransformer(original):
        def __init__(self):
            super().__init__()
            # Preserve the archived backbone initialization and RNG schedule.
            state = torch.get_rng_state()
            self.patch_proj = ModalityRawProjection(layout, base.D_MODEL)
            torch.set_rng_state(state)

    base.SpectraConditionTransformer = RawPatchTransformer
    return RawPatchTransformer


def make_loop_class(base, layout):
    class RawPatchClosedLoop(base.TransformerClosedLoop):
        def __init__(self):
            self.raw_cache = {}
            self.iteration_details = {}
            self.iteration_number = 0
            super().__init__()
            self.fixed_target_hash = self.target_hash()
            save_json(base.OUT_ROOT / "raw_patch_layout.json", layout.metadata())

        def target_hash(self):
            value = hashlib.sha256()
            for modality in base.MODALITIES:
                for array in self.target_spectra[modality]:
                    value.update(np.ascontiguousarray(array).tobytes())
            return value.hexdigest()

        def raw_features(self, reaction):
            if reaction not in self.raw_cache:
                self.raw_cache[reaction] = layout.encode(self.spectra[reaction])
            return self.raw_cache[reaction]

        def build_arrays(self):
            reactions = sorted(self.current)
            records, conditions, patches, labels, valid, metric_valid = [], [], [], [], [], []
            for reaction in reactions:
                record = self.current[reaction]
                self.ensure_spectra(record)
                distances = self.spectral_distances(reaction)
                metric = base.metric_vector(record)
                records.append(record)
                conditions.append(base.condition_vector(record))
                patches.append(self.raw_features(reaction))
                labels.append([math.log1p(distances.get("GPC", 0.0)),
                               math.log1p(distances.get("NMR", 0.0)),
                               math.log1p(distances.get("IR", 0.0)),
                               math.log1p(sum(distances.values())), math.log1p(metric[0]),
                               metric[1], metric[2] / 100.0])
                valid.append(record["Suffix"] in base.VALID_SUFFIXES)
                metric_valid.append(base.is_metric_valid(record))
            return (reactions, records, np.stack(conditions).astype(np.float32),
                    np.stack(patches), np.array(labels, dtype=np.float32),
                    np.array(valid, dtype=bool), np.array(metric_valid, dtype=bool))

        def candidate_tokens(self, candidate_rows, norm):
            cond = np.vstack([base.condition_vector(row) for row in candidate_rows]).astype(np.float32)
            cond_z = base.standardize_apply(cond, norm["condition_center"], norm["condition_scale"])
            current_reactions = sorted(self.current)
            current_cond = np.vstack([base.condition_vector(self.current[r]) for r in current_reactions])
            current_patch = np.stack([self.raw_features(r) for r in current_reactions])
            center, scale = base.standardize_fit(current_cond)
            current_z = base.standardize_apply(current_cond, center, scale)
            candidate_z = base.standardize_apply(cond, center, scale)
            estimated = np.empty((len(candidate_rows), layout.flat_dim), dtype=np.float32)
            for index, z in enumerate(candidate_z):
                distances = np.linalg.norm(current_z - z, axis=1)
                order = np.argsort(distances)[:8]
                weights = np.exp(-distances[order] / max(float(np.median(distances[order])), 1e-6))
                weights = weights / weights.sum()
                estimated[index] = np.tensordot(weights, current_patch[order], axes=(0, 0))
            estimated -= norm["patch_center"]
            estimated /= norm["patch_scale"]
            if not np.isfinite(estimated).all():
                raise ValueError("Non-finite candidate raw-patch features.")
            return cond_z, estimated

        def train_one_round(self, iteration):
            self.iteration_number = iteration
            self.round_started = time.monotonic()
            print("TRAIN_START seed={} round={} states={} epochs={} raw_points={}".format(
                base.SEED, iteration, len(self.current), base.EPOCHS_PER_ROUND, layout.flat_dim), flush=True)
            model, norm, train_loss, valid_loss = super().train_one_round(iteration)
            checkpoint_path = base.OUT_ROOT / "models/round_{:03d}.pt".format(iteration)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["model_version"] = VERSION
            checkpoint["raw_patch_layout"] = layout.metadata()
            checkpoint["output_names"] = OUTPUT_NAMES
            checkpoint["engine_source_sha256"] = digest(SOURCE)
            torch.save(checkpoint, checkpoint_path)
            return model, norm, train_loss, valid_loss

        def recommend_next_batch(self, model, norm):
            result = super().recommend_next_batch(model, norm)
            # Reveal measurements only after the property scorer selects a batch.
            record = result["best_row"]
            self.ensure_spectra(record)
            distances = self.spectral_distances(record["Reaction"])
            measured = base.metric_vector(record)
            if self.target_hash() != self.fixed_target_hash:
                raise RuntimeError("Fixed target spectra changed during the run.")
            self.iteration_details[self.iteration_number] = {
                "Model_Version": VERSION, "Candidate_Input_Mode": "8_neighbors_raw_intensity_patches",
                "Actual_Mw": float(measured[0]), "Actual_PDI": float(measured[1]),
                "Actual_Conversion": float(measured[2]), "Actual_GPC_Distance": distances["GPC"],
                "Actual_NMR_Distance": distances["NMR"], "Actual_FTIR_Distance": distances["IR"],
                "Actual_Aux_Distance": sum(distances.values()),
                "Round_Wall_Seconds": time.monotonic() - self.round_started,
                "Fixed_Target_Hash": self.fixed_target_hash,
            }
            return result

        def write_logs(self, logs, stop_reason):
            for row in logs:
                row.update(self.iteration_details.get(row["Iteration"], {}))
            super().write_logs(logs, stop_reason)
            path = base.OUT_ROOT / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary.update({"model": VERSION, "variant_scope": "complete raw-intensity patches and learned modality-specific CNN encoders",
                            "execution_kind": base.RUNTIME_CONFIG["execution_kind"],
                            "training_input": "measured raw-intensity patches plus matched conditions",
                            "candidate_input": "8 condition-nearest revealed samples estimate complete raw-intensity patches",
                            "target_spectra_in_candidate_input": False,
                            "target_spectra_role": "fixed spectral-distance supervision labels only",
                            "candidate_neighbors": 8, "spectral_ranking_weight": 0,
                            "patch_encoder": {"type": "modality-specific 1D CNN, shared over patches within a modality",
                                              "channels": [1, 16, 32], "kernels": [7, 5], "strides": [4, 4],
                                              "paddings": [3, 2], "activation": "GELU", "token_dim": base.D_MODEL,
                                              "joint_training": True, "handcrafted_statistics": False},
                            "output_names": OUTPUT_NAMES, "raw_patch_layout": layout.metadata(),
                            "patching": {m: "{} raw patches, padded width {}".format(e["patches"], e["patch_width"])
                                         for m, e in layout.entries.items()},
                            "fixed_target_hash": self.fixed_target_hash, "engine_source_sha256": digest(SOURCE),
                            "raw_patch_source_sha256": digest(__file__), "epochs_per_round": base.EPOCHS_PER_ROUND,
                            "max_rounds": base.MAX_ROUNDS,
                            "training_strategy": "fixed epoch budget; restore lowest weighted validation loss checkpoint; no early stopping",
                            "optimizer": {"name": "AdamW", "learning_rate": base.LEARNING_RATE,
                                          "weight_decay": base.WEIGHT_DECAY, "batch_size": base.BATCH_SIZE},
                            "initial_workbook_sha256": digest(base.INITIAL_XLSX),
                            "all_workbook_sha256": digest(base.ALL_XLSX)})
            save_json(path, summary)

    return RawPatchClosedLoop


def validate_workbooks(base):
    initial = base.read_workbook(base.INITIAL_XLSX, "Reaction")
    all_rows = base.read_workbook(base.ALL_XLSX, "Reaction_ID")
    numeric = ["Total_mols", "Initiator", "Time", "Temp", "Mw", "PDI", "Conc_Mean"]
    for label, rows in [("initial", initial), ("database", all_rows)]:
        if not rows or len({r["Reaction"] for r in rows}) != len(rows):
            raise ValueError(label + " workbook must contain unique, nonempty reaction IDs.")
        batches = {}
        for row in rows:
            if row["Reaction"] != "{}-{:02d}".format(row["Batch"], row["Suffix"]):
                raise ValueError("Expected batch-01 through batch-12 IDs: " + row["Reaction"])
            for key in numeric:
                try:
                    finite = np.isfinite(float(row[key]))
                except (KeyError, TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError("Missing/non-numeric {} for {}".format(key, row["Reaction"]))
            batches.setdefault(row["Batch"], []).append(row)
        for batch, members in batches.items():
            if len(members) != 12 or {r["Suffix"] for r in members} != set(range(1, 13)):
                raise ValueError("Every batch must contain all 12 states: " + batch)
            for row in members:
                if float(row["Time"]) != 120 * row["Suffix"]:
                    raise ValueError("Expected sampling at 120, 240, ..., 1440 min: " + row["Reaction"])
            for key in ["Total_mols", "Initiator", "Temp"]:
                if len({float(r[key]) for r in members}) != 1:
                    raise ValueError("Inconsistent batch condition {}: {}".format(key, batch))
    lookup = {row["Reaction"]: row for row in all_rows}
    for row in initial:
        if row["Reaction"] not in lookup:
            raise ValueError("Initial reaction absent from database: " + row["Reaction"])
    return initial, all_rows


def preflight(base, layout, spectra_root, full_audit=False):
    initial, all_rows = validate_workbooks(base)
    expected = {row["Reaction"] for row in all_rows}
    lookup = {row["Reaction"]: row for row in all_rows}
    mismatches = []
    for row in initial:
        fields = [key for key in ["Total_mols", "Initiator", "Time", "Temp", "Mw", "PDI", "Conc_Mean"]
                  if not np.isclose(float(row[key]), float(lookup[row["Reaction"]][key]), rtol=1e-6, atol=1e-6)]
        if fields:
            mismatches.append({"reaction": row["Reaction"], "fields": fields})
    if mismatches:
        print("INITIAL_DATABASE_DIFFERENCES {} states; full-database values take precedence".format(
            len(mismatches)), flush=True)
    audited, extras = {}, {}
    samples = sorted(expected if full_audit else {r["Reaction"] for r in initial})
    for modality in base.MODALITIES:
        folder = Path(spectra_root) / DIRECTORIES[modality]
        actual = {p.stem for p in folder.glob("*.csv")}
        missing = expected - actual
        if missing:
            raise ValueError("Missing {} spectra ({}): {}".format(
                modality, len(missing), ", ".join(sorted(missing)[:5])))
        extras[DIRECTORIES[modality]] = len(actual - expected)
        for reaction in samples:
            layout.validate(modality, *base.read_spectrum(modality, reaction))
        audited[DIRECTORIES[modality]] = len(samples)
        print("RAW_GRID_VERIFIED {}: {} spectra; {} points; no resampling".format(
            modality, len(samples), layout.entries[modality]["points"]), flush=True)
    return {"initial_states": len(initial), "initial_batches": len({r["Batch"] for r in initial}),
            "database_states": len(all_rows), "database_batches": len({r["Batch"] for r in all_rows}),
            "grid_audited": audited, "extra_spectra_ignored": extras,
            "initial_database_value_mismatches": mismatches,
            "replay_value_source": "All workbook; Initial workbook selects revealed reaction IDs",
            "raw_patch_layout": layout.metadata(), "output_dim": 7, "candidate_neighbors": 8,
            "spectral_distance_supervision": True, "spectral_ranking_weight": 0,
            "initial_target_satisfying_states": sum(bool(base.is_success(lookup[r["Reaction"]])) for r in initial)}
