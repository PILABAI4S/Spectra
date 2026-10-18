"""SPA-Former program and portable spectral state navigation adapter."""

import csv
import hashlib
import importlib.util
import json
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

from .seeds import validate_seeds

VERSION = "spectral state navigation"
SOURCE = Path(__file__).with_name("_engine.py")
FOLDERS = {"GPC": "GPC", "NMR": "NMR", "IR": "FTIR"}
OUTPUT_NAMES = ["D_GPC", "D_NMR", "D_FTIR", "D_aux", "Mw", "PDI", "Conversion"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configure(seed, args):
    seed = validate_seeds([seed])[0]
    settings = {"initial_file": str(Path(args.initial_file).resolve()),
                "all_file": str(Path(args.all_file).resolve()),
                "spectra_root": str(Path(args.spectra_root).resolve()),
                "run_directory": str(Path(args.output).resolve() / "run_seed_{}".format(seed)),
                "seed": seed, "max_rounds": args.max_rounds, "epochs": args.epochs,
                "execution_kind": "smoke" if getattr(args, "smoke", False) else "replay"}
    name = "spectral_state_navigation_runtime_{}".format(seed)
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    base = importlib.util.module_from_spec(spec)
    base.RUNTIME_CONFIG = settings
    sys.modules[name] = base
    spec.loader.exec_module(base)

    def read_spectrum(modality, reaction):
        path = Path(args.spectra_root) / FOLDERS[modality] / (reaction + ".csv")
        kwargs = {} if modality == "NMR" else {"delimiter": ",", "skiprows": 1}
        array = np.loadtxt(path, dtype=np.float32, **kwargs)
        if (array.ndim != 2 or array.shape[1] != 2 or len(array) < base.PATCH_COUNT[modality]
                or not np.isfinite(array).all()):
            raise ValueError("Invalid measured spectrum: " + str(path))
        delta = np.diff(array[:, 0].astype(np.float64))
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError("Spectral coordinate must be strictly monotonic: " + str(path))
        return array[:, 0], array[:, 1]

    base.read_spectrum = read_spectrum
    return base, settings


def validate_workbooks(base):
    initial = base.read_workbook(base.INITIAL_XLSX, "Reaction")
    rows = base.read_workbook(base.ALL_XLSX, "Reaction_ID")
    numeric = ["Total_mols", "Initiator", "Time", "Temp", "Mw", "PDI", "Conc_Mean"]
    for label, values in [("initial", initial), ("database", rows)]:
        if not values or len({v["Reaction"] for v in values}) != len(values):
            raise ValueError(label + " workbook must contain unique nonempty IDs")
        batches = {}
        for record in values:
            if record["Reaction"] != "{}-{:02d}".format(record["Batch"], record["Suffix"]):
                raise ValueError("Invalid sample identifier: " + record["Reaction"])
            for key in numeric:
                try:
                    finite = np.isfinite(float(record[key]))
                except (KeyError, TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError("Non-finite or missing {} for {}".format(key, record["Reaction"]))
            batches.setdefault(record["Batch"], []).append(record)
        for batch, members in batches.items():
            if len(members) != 12 or {v["Suffix"] for v in members} != set(range(1, 13)):
                raise ValueError("Require all 12 time states for batch " + batch)
            if any(float(v["Time"]) != 120 * v["Suffix"] for v in members):
                raise ValueError("Require sampling at 120, 240, ..., 1440 min: " + batch)
            for key in ["Total_mols", "Initiator", "Temp"]:
                if len({float(v[key]) for v in members}) != 1:
                    raise ValueError("Inconsistent batch condition: " + batch + " / " + key)
    lookup = {v["Reaction"]: v for v in rows}
    if any(v["Reaction"] not in lookup for v in initial):
        raise ValueError("Initial sample absent from full database")
    return initial, rows


def preflight(base, spectra_root, full_audit=False):
    initial, rows = validate_workbooks(base)
    expected = {v["Reaction"] for v in rows}
    lookup = {v["Reaction"]: v for v in rows}
    mismatches = []
    for record in initial:
        changed = [key for key in ["Total_mols", "Initiator", "Time", "Temp", "Mw", "PDI", "Conc_Mean"]
                   if not np.isclose(float(record[key]), float(lookup[record["Reaction"]][key]), rtol=1e-6, atol=1e-6)]
        if changed:
            mismatches.append({"reaction": record["Reaction"], "fields": changed})
    audited, extras = {}, {}
    for modality, folder in FOLDERS.items():
        available = {p.stem for p in (Path(spectra_root) / folder).glob("*.csv")}
        missing = expected - available
        if missing:
            raise ValueError("Missing {} spectra: {}".format(folder, ", ".join(sorted(missing)[:5])))
        extras[folder] = len(available - expected)
        samples = sorted(expected if full_audit else {v["Reaction"] for v in initial})
        for reaction in samples:
            base.read_spectrum(modality, reaction)
        audited[folder] = len(samples)
    if mismatches:
        print("INITIAL_DATABASE_DIFFERENCES {}; All workbook values take precedence".format(len(mismatches)), flush=True)
    return {"model": VERSION, "database_states": len(rows), "database_batches": len({v["Batch"] for v in rows}),
            "initial_states": len(initial), "initial_batches": len({v["Batch"] for v in initial}),
            "spectra_audited": audited, "extra_spectra_ignored": extras,
            "initial_database_value_mismatches": mismatches, "replay_values_source": "All workbook",
            "initial_target_satisfying_states": int(sum(base.is_success(lookup[v["Reaction"]]) for v in initial)),
            "property_loss_masked_states": sum(not base.is_metric_valid(v) for v in rows),
            "patch_shape": [60, 5], "output_dim": 7, "CNN": False,
            "candidate_neighbors": 8, "spectral_distance_supervision": True, "spectral_ranking_weight": 0}


def make_loop_class(base):
    class NavigationClosedLoop(base.TransformerClosedLoop):
        def __init__(self):
            self.capture_epochs, self.details = False, {}
            super().__init__()
            self.fixed_target_hash = self.target_hash()

        def target_hash(self):
            value = hashlib.sha256()
            for modality in base.MODALITIES:
                for array in self.target_spectra[modality]:
                    value.update(np.ascontiguousarray(array).tobytes())
            return value.hexdigest()

        def multitask_loss(self, pred, target, mask):
            loss = super().multitask_loss(pred, target, mask)
            if self.capture_epochs:
                value = float(loss.detach().cpu())
                if torch.is_grad_enabled():
                    self.train_losses.append(value)
                else:
                    self.epoch_rows.append({"Epoch": len(self.epoch_rows) + 1,
                                            "Training_loss": float(np.mean(self.train_losses)),
                                            "Validation_loss": value})
                    self.train_losses = []
            return loss

        def train_one_round(self, iteration):
            self.iteration, self.round_start = iteration, time.monotonic()
            self.epoch_rows, self.train_losses, self.capture_epochs = [], [], True
            try:
                result = super().train_one_round(iteration)
            finally:
                self.capture_epochs = False
            if len(self.epoch_rows) != base.EPOCHS_PER_ROUND:
                raise RuntimeError("Incomplete epoch history")
            self.best_epoch = min(self.epoch_rows, key=lambda v: v["Validation_loss"])["Epoch"]
            if min(v["Validation_loss"] for v in self.epoch_rows) != result[3]:
                raise RuntimeError("Checkpoint loss differs from minimum validation loss")
            with (base.OUT_ROOT / "epochs_round_{:03d}.csv".format(iteration)).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.epoch_rows[0]))
                writer.writeheader()
                writer.writerows(self.epoch_rows)
            return result

        def recommend_next_batch(self, model, norm):
            result = super().recommend_next_batch(model, norm)
            record = result["best_row"]
            self.ensure_spectra(record)
            distance = self.spectral_distances(record["Reaction"])
            measured = base.metric_vector(record)
            if self.target_hash() != self.fixed_target_hash:
                raise RuntimeError("Fixed target changed during the run")
            self.details[self.iteration] = {
                "Actual_Mw": float(measured[0]), "Actual_PDI": float(measured[1]),
                "Actual_Conversion": float(measured[2]), "Actual_GPC_Distance": distance["GPC"],
                "Actual_NMR_Distance": distance["NMR"], "Actual_FTIR_Distance": distance["IR"],
                "Actual_Aux_Distance": sum(distance.values()), "Best_epoch": self.best_epoch,
                "Round_Wall_Seconds": time.monotonic() - self.round_start,
                "Fixed_Target_Hash": self.fixed_target_hash}
            return result

        def write_logs(self, logs, reason):
            for row in logs:
                row.update(self.details.get(row["Iteration"], {}))
            super().write_logs(logs, reason)
            path = base.OUT_ROOT / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value.update(model=VERSION, output_names=OUTPUT_NAMES, patch_encoder="five intensity statistics; Linear(5,96)",
                         CNN=False, spectral_distance_supervision=True, spectral_ranking_weight=0,
                         candidate_neighbors=8, target_spectra_in_candidate_input=False,
                         target_spectra_role="fixed spectral-distance supervision labels",
                         fixed_target_hash=self.fixed_target_hash, epochs_per_round=base.EPOCHS_PER_ROUND,
                         max_rounds=base.MAX_ROUNDS, execution_kind=base.RUNTIME_CONFIG["execution_kind"],
                         database_states=len(self.all_rows), database_batches=len(self.all_by_batch),
                         training_strategy="fixed epoch budget; minimum validation-loss checkpoint; no early stopping",
                         optimizer={"name": "AdamW", "learning_rate": base.LEARNING_RATE,
                                    "weight_decay": base.WEIGHT_DECAY, "batch_size": base.BATCH_SIZE},
                         engine_source_sha256=digest(SOURCE), adapter_source_sha256=digest(__file__),
                         all_workbook_sha256=digest(base.ALL_XLSX), initial_workbook_sha256=digest(base.INITIAL_XLSX))
            save_json(path, value)

    return NavigationClosedLoop
