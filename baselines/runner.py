"""Single isolated baseline task; invoked by the shared benchmark entry point."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from benchmark.data import is_hit, load_data
from benchmark.io import clean, save, sha, write_csv
from spa_former.seeds import read_manifest
from .registry import METHODS
from .runtime import prepare


def audit(logs, reason, initial, lookup, batches, cap):
    if not logs or len(logs) > cap:
        raise RuntimeError("Empty or over-budget trajectory")
    seen = {r["Batch"] for r in initial}
    rows = []
    for iteration, row in enumerate(logs, 1):
        sample, batch = row["Recommended_Reaction"], row["Recommended_Batch"]
        if row["Iteration"] != iteration or batch in seen or lookup[sample]["Batch"] != batch:
            raise RuntimeError("Invalid or duplicate recommended batch")
        seen.add(batch)
        hits = {r["Reaction"] for r in batches[batch] if is_hit(r)}
        count = row.get("Batch_success_count", row.get("Batch_Success_Count"))
        if int(count) != len(hits) or set(filter(None, row["Success_Reactions"].split(";"))) != hits:
            raise RuntimeError("Batch success disagrees with measured outcomes")
        if hits and iteration != len(logs):
            raise RuntimeError("Loop continued after target attainment")
        actual = lookup[sample]
        rows.append({**row, "Actual_Mw": actual["Mw"], "Actual_PDI": actual["PDI"],
                     "Actual_Conversion": actual["Conc_Mean"], "Actual_Total_mols": actual["Total_mols"],
                     "Actual_Initiator": actual["Initiator"], "Actual_Time_min": actual["Time"],
                     "Actual_Temperature_C": actual["Temp"]})
    success = any(is_hit(r) for r in batches[logs[-1]["Recommended_Batch"]])
    if (reason == "success_condition_met") != success or (not success and len(logs) != cap):
        raise RuntimeError("Stopping rule disagrees with measured target/cap")
    return rows, success


def versions():
    result = {"python": platform.python_version()}
    for name in ["numpy", "scipy", "scikit-learn", "torch", "tabpfn", "openpyxl"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    seed = read_manifest(args.seeds_file)[args.seed_index]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Baseline output must be empty: " + str(out))
    initial, records, lookup, batches, checked = load_data(args.initial_file, args.all_file)
    if checked["initial_target_satisfying_states"]:
        raise ValueError("Initial design already satisfies the target")
    if checked["database_batches"] - checked["initial_batches"] < args.max_rounds and not args.check_only:
        raise ValueError("Insufficient unrevealed batches for the requested cap")
    if args.method in ["descriptor_transformer", "tabpfn"]:
        import torch
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
    module, base, config = prepare(args.method, seed, args.initial_file, args.all_file, out,
                                    args.max_rounds, args.epochs, args.device)
    save(out / "preflight.json", checked)
    save(out / "run_settings.json", {"method": args.method, "seed": seed, "repeat": args.seed_index + 1,
         "configuration": config, "software": versions(), "all_workbook_sha256": sha(args.all_file),
         "initial_workbook_sha256": sha(args.initial_file), "local_seed_manifest_sha256": sha(args.seeds_file),
         "source_sha256": {p.name: sha(p) for p in Path(__file__).parent.joinpath("engines").glob("*.py")},
         "preprocessing": "Original method-specific preprocessing retained"})
    if args.check_only:
        print("CHECK_PASSED " + args.method, flush=True)
        return
    started = time.monotonic()
    save(out / "progress.json", {"state": "running", "method": args.method, "seed": seed})
    try:
        if args.method.startswith("descriptor_bo_"):
            logs, reason = module.PropertyScalarBO().run()
            native = json.loads((out / "bo_property_scalar_summary.json").read_text())
        else:
            if args.method == "descriptor_transformer":
                base.set_seed(seed)
            native, logs = base.run_one(seed, initial, records, lookup, batches)
            reason = native["Stop_reason"]
        prepared, success = audit(logs, reason, initial, lookup, batches, args.max_rounds)
        write_csv(out / "native_iteration_log.csv", logs)
        write_csv(out / "iteration_log.csv", prepared)
        save(out / "native_summary.json", native)
        save(out / "summary.json", {"state": "completed", "method": args.method, "seed": seed,
             "repeat": args.seed_index + 1, "rounds": len(logs), "success": success, "stop_reason": reason,
             "max_rounds": args.max_rounds, "execution_kind": "smoke" if args.smoke else "replay",
             "database_batches": len(batches), "database_states": len(records),
             "final_batch": logs[-1]["Recommended_Batch"], "successful_samples": logs[-1]["Success_Reactions"],
             "wall_seconds": time.monotonic() - started})
        save(out / "progress.json", {"state": "completed", "rounds": len(logs), "success": success})
    except Exception as error:
        save(out / "progress.json", {"state": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--initial-file", type=Path, required=True)
    parser.add_argument("--all-file", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_rounds <= 200 or min(args.epochs, args.threads) < 1:
        parser.error("Require1-200 rounds and positive epochs/threads")
    if not 0 <= args.seed_index < len(read_manifest(args.seeds_file)):
        parser.error("Invalid manifest index")
    if args.smoke:
        args.max_rounds, args.epochs = min(args.max_rounds, 2), min(args.epochs, 3)
    run(args)


if __name__ == "__main__":
    main()
