"""Run selected methods with one freshly generated paired seed manifest."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

import numpy as np

from baselines.registry import ALL_METHODS, METHODS
from spa_former.seeds import make_manifest, read_manifest, write_manifest
from .data import load_data
from .io import save, write_csv
from .seeds import generate_paired_seeds, validate_paired_seeds


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--methods", nargs="+", choices=["all", *ALL_METHODS], default=["all"])
    result.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Dataset directory; defaults to the project directory.")
    result.add_argument("--initial-file", type=Path)
    result.add_argument("--all-file", type=Path)
    result.add_argument("--spectra-root", type=Path)
    group = result.add_mutually_exclusive_group()
    group.add_argument("--repetitions", type=int)
    group.add_argument("--seeds-file", type=Path, help="Optional private manifest, never bundled")
    result.add_argument("--seed-index", type=int, help="Optional Slurm task index in the generated manifest")
    result.add_argument("--output", type=Path)
    result.add_argument("--max-rounds", type=int, default=200)
    result.add_argument("--epochs", type=int, default=180)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--audit-spectra", action="store_true")
    result.add_argument("--smoke", action="store_true")
    return result


def collect(output, methods):
    rows = []
    for method in methods:
        method_rows = []
        for trial in sorted((output / method).glob("repeat_*")):
            if method == "spectral_state_navigation":
                paths = list(trial.glob("run_seed_*/summary.json"))
            else:
                paths = list(trial.glob("summary.json"))
            for path in paths:
                value = json.loads(path.read_text())
                if value["stop_reason"] == "running" or value.get("state") == "failed":
                    continue
                repeat = int(trial.name.split("_")[1])
                method_rows.append({"Method": method, "Repeat": repeat, "Seed": value["seed"],
                                    "Iterations": value["rounds"], "Success": value["stop_reason"] == "success_condition_met",
                                    "Stop_reason": value["stop_reason"], "Max_rounds": value["max_rounds"]})
        write_csv(output / method / "run_summary.csv", method_rows)
        rows.extend(method_rows)
    write_csv(output / "run_summary.csv", rows)
    stats = []
    for method in methods:
        selected = [r for r in rows if r["Method"] == method]
        if selected:
            counts = np.array([r["Iterations"] for r in selected])
            stats.append({"Method": method, "Completed": len(selected), "Successful": sum(r["Success"] for r in selected),
                          "Mean": float(counts.mean()), "Sample_SD": float(counts.std(ddof=1)) if len(counts) > 1 else None,
                          "Median": float(np.median(counts)), "Min": int(counts.min()), "Max": int(counts.max())})
    write_csv(output / "statistics.csv", stats)
    return rows


def execute(args):
    methods = list(ALL_METHODS) if "all" in args.methods else list(args.methods)
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate selected methods")
    if not 1 <= args.max_rounds <= 200 or min(args.epochs, args.threads) < 1:
        raise ValueError("Require1-200 rounds and positive epochs/threads")
    if args.seed_index is not None and args.seeds_file is None:
        raise ValueError("--seed-index requires --seeds-file")
    seeds = (validate_paired_seeds(read_manifest(args.seeds_file)) if args.seeds_file
             else generate_paired_seeds(20 if args.repetitions is None else args.repetitions))
    if args.seed_index is not None and not 0 <= args.seed_index < len(seeds):
        raise ValueError("Invalid manifest index")
    indices = [args.seed_index] if args.seed_index is not None else list(range(len(seeds)))
    if args.check_only:
        indices = indices[:1]
    initial_file = (args.initial_file or args.data_dir / "Initial_experiments.xlsx").resolve()
    all_file = (args.all_file or args.data_dir / "All_experiments.xlsx").resolve()
    spectra = (args.spectra_root or args.data_dir).resolve()
    _, _, _, _, checked = load_data(initial_file, all_file)
    if args.smoke:
        args.max_rounds, args.epochs = min(args.max_rounds, 2), min(args.epochs, 3)
    if not args.check_only:
        if checked["initial_target_satisfying_states"]:
            raise ValueError("Initial design already satisfies target")
        if checked["database_batches"] - checked["initial_batches"] < args.max_rounds:
            raise ValueError("Insufficient unrevealed batches for requested cap")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("results") / ("comparison_" + stamp + "_" + secrets.token_hex(4))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Benchmark output must be new or empty")
    write_manifest(out / "seed_manifest.json", make_manifest(seeds,
                   "replayed private generated manifest" if args.seeds_file else "secrets.randbits(32); unique; estimator offset reserve"))
    save(out / "run_config.json", {"methods": methods, "max_rounds": args.max_rounds, "epochs": args.epochs,
         "paired_seed_count": len(seeds), "selected_indices": indices, "preflight": checked,
         "execution_kind": "preflight" if args.check_only else "smoke" if args.smoke else "replay"})
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    project = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = project + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for index in indices:
        for method in methods:
            trial = out / method / "repeat_{:03d}".format(index + 1)
            trial.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-B", "-m", "spa_former" if method == "spectral_state_navigation" else "baselines.runner",
                   "--initial-file", str(initial_file), "--all-file", str(all_file),
                   "--seeds-file", str(out / "seed_manifest.json"), "--seed-index", str(index),
                   "--output", str(trial), "--max-rounds", str(args.max_rounds), "--epochs", str(args.epochs),
                   "--threads", str(args.threads), "--device", args.device]
            if method == "spectral_state_navigation":
                cmd += ["--spectra-root", str(spectra)]
                if args.audit_spectra:
                    cmd += ["--audit-spectra"]
            else:
                cmd += ["--method", method]
            if args.smoke:
                cmd += ["--smoke"]
            if args.check_only:
                cmd += ["--check-only"]
            log_path = trial.parent / (trial.name + ".log")
            print("TASK_START {} repeat={}".format(method, index + 1), flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                save(out / "progress.json", {"state": "failed", "method": method, "repeat": index + 1, "log": str(log_path)})
                raise RuntimeError("Task failed; inspect " + str(log_path))
            if not args.check_only:
                collect(out, methods)
    save(out / "progress.json", {"state": "completed", "tasks": len(indices) * len(methods)})
    return out


def main(argv=None):
    try:
        output = execute(parser().parse_args(argv))
        print("OUTPUT " + str(output), flush=True)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit("ERROR: " + str(error))


if __name__ == "__main__":
    main()
