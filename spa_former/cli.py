"""Spectral state navigation database replay with code-generated run seeds."""

import argparse
import csv
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import platform
import secrets

from . import __version__
from .seeds import generate_seeds, make_manifest, read_manifest, validate_seeds, write_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Dataset directory; defaults to the project directory.")
    parser.add_argument("--initial-file", type=Path)
    parser.add_argument("--all-file", type=Path)
    parser.add_argument("--spectra-root", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--repetitions", type=int, help="Generate fresh seeds; default: 20 runs.")
    group.add_argument("--seeds-file", type=Path, help="Replay a locally generated manifest; none is supplied.")
    parser.add_argument("--seed-index", type=int, help="Select one manifest entry for a Slurm array.")
    parser.add_argument("--output", type=Path, help="New or empty local output directory.")
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--audit-spectra", action="store_true", help="Parse every supplied spectrum before training.")
    parser.add_argument("--smoke", action="store_true", help="At most2 rounds and3 epochs; not a scientific result.")
    return parser


def choose_seeds(args):
    if args.seed_index is not None and args.seeds_file is None:
        raise ValueError("--seed-index requires a locally generated --seeds-file")
    if args.seeds_file is not None:
        seeds = read_manifest(args.seeds_file)
        if args.seed_index is not None:
            if not 0 <= args.seed_index < len(seeds):
                raise ValueError("Seed index outside manifest")
            seeds = [seeds[args.seed_index]]
        return seeds, "replayed local generated manifest"
    return generate_seeds(20 if args.repetitions is None else args.repetitions), "secrets.randbits(32); unique"


def claim_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("Output directory must be new or empty: " + str(path))
    return path


def navigation_module():
    return import_module(".spa-former", package=__package__)


def aggregate(output, seeds):
    import numpy as np

    navigation = navigation_module()
    VERSION, save_json = navigation.VERSION, navigation.save_json

    rows = []
    for seed in seeds:
        path = output / "run_seed_{}".format(seed) / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["stop_reason"] == "running":
            continue
        final = summary.get("final") or {}
        rows.append({"Model": VERSION, "Seed": seed, "Iterations": summary["rounds"],
                     "Target_reached": summary["stop_reason"] == "success_condition_met",
                     "Max_rounds": summary["max_rounds"], "Epochs_per_round": summary["epochs_per_round"],
                     "Execution_kind": summary["execution_kind"], "Stop_reason": summary["stop_reason"],
                     "Final_batch": final.get("Recommended_Batch", ""), "Successful_samples": final.get("Success_Reactions", "")})
    if rows:
        with (output / "run_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = np.array([v["Iterations"] for v in rows], dtype=np.float64)
        save_json(output / "statistics.json", {"completed_runs": len(rows), "requested_runs": len(seeds),
                  "successful_runs": sum(v["Target_reached"] for v in rows), "mean_iterations": float(counts.mean()),
                  "median_iterations": float(np.median(counts)), "sample_sd_iterations": float(counts.std(ddof=1)) if len(rows) > 1 else None,
                  "min_iterations": int(counts.min()), "max_iterations": int(counts.max()),
                  "trimming": "none; capped failures included at the cap"})
    return rows


def execute(args):
    navigation = navigation_module()
    import numpy as np
    import openpyxl
    import torch

    if not 1 <= args.max_rounds <= 200 or min(args.epochs, args.threads) < 1:
        raise ValueError("Require 1-200 rounds and positive epochs/threads")
    seeds, policy = choose_seeds(args)
    args.initial_file = (args.initial_file or args.data_dir / "Initial_experiments_sol.xlsx").resolve()
    args.all_file = (args.all_file or args.data_dir / "All_experiments_sol.xlsx").resolve()
    args.spectra_root = (args.spectra_root or args.data_dir).resolve()
    for path in [args.initial_file, args.all_file]:
        if not path.is_file():
            raise FileNotFoundError("Required experiment workbook not found: " + str(path))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.smoke:
        args.max_rounds, args.epochs = min(args.max_rounds, 2), min(args.epochs, 3)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("results") / (stamp + "_" + secrets.token_hex(4))
    args.output = claim_output(args.output)
    manifest = make_manifest(seeds, policy)
    manifest["execution_kind"] = "preflight" if args.check_only else "smoke" if args.smoke else "replay"
    write_manifest(args.output / "seed_manifest.json", manifest)
    torch.set_num_threads(args.threads)
    base, _ = navigation.configure(seeds[0], args)
    navigation.validate_workbooks(base)
    layout = navigation.fit_layout(base)
    checked = navigation.preflight(base, layout, args.spectra_root, args.audit_spectra)
    base.set_seed(seeds[0])
    model = navigation.install_raw_model(base, layout)().eval()
    checked["parameter_count"] = sum(p.numel() for p in model.parameters())
    checked.update(model=navigation.VERSION, CNN=True, patch_encoder="modality-specific 1D CNN")
    with torch.no_grad():
        prediction = model(torch.zeros(2, layout.flat_dim), torch.zeros(2, 4))
    if prediction.shape != (2, 7) or not torch.isfinite(prediction).all():
        raise RuntimeError("Unexpected spectral state navigation architecture")
    del model
    navigation.save_json(args.output / "preflight.json", checked)
    navigation.save_json(args.output / "provenance.json", {"package_version": __version__, "python": platform.python_version(),
                 "numpy": np.__version__, "openpyxl": openpyxl.__version__, "torch": torch.__version__,
                 "cuda_runtime": torch.version.cuda, "deterministic_algorithms_forced": False,
                 "all_workbook_sha256": navigation.digest(args.all_file), "initial_workbook_sha256": navigation.digest(args.initial_file),
                 "package_source_sha256": {p.name: navigation.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})
    if args.check_only:
        print("CHECK_PASSED " + str(args.output), flush=True)
        return args.output
    if checked["initial_target_satisfying_states"]:
        raise ValueError("Choose an initial design with no existing target-satisfying states")
    if checked["database_batches"] - checked["initial_batches"] < args.max_rounds:
        raise ValueError("Fewer unrevealed batches than requested maximum rounds; reduce --max-rounds")
    for seed in seeds:
        base, settings = navigation.configure(seed, args)
        layout = navigation.fit_layout(base)
        navigation.install_raw_model(base, layout)
        base.OUT_ROOT.mkdir(exist_ok=False)
        loop = navigation.make_loop_class(base, layout)()
        if args.device != "auto":
            loop.device = torch.device(args.device)
        navigation.save_json(base.OUT_ROOT / "run_settings.json", {"model": navigation.VERSION, "seed_generation": policy,
                     "configuration": settings, "output_dim": 7, "candidate_neighbors": 8,
                     "spectral_distance_supervision": True, "spectral_ranking_weight": 0, "CNN": True,
                     "target_spectra_in_candidate_input": False, "raw_patch_layout": layout.metadata(),
                     "patch_encoder": {"channels": [1, 16, 32], "kernels": [7, 5], "strides": [4, 4],
                                       "paddings": [3, 2], "activation": "GELU", "token_dim": 96}})
        print("RUN_START model={} seed={} device={}".format(navigation.VERSION, seed, loop.device), flush=True)
        try:
            logs, reason = loop.run()
        except Exception as error:
            navigation.save_json(base.OUT_ROOT / "error.json", {"type": type(error).__name__, "message": str(error)})
            raise
        aggregate(args.output, seeds)
        print("RUN_END rounds={} reason={}".format(len(logs), reason), flush=True)
        del loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return args.output


def main(argv=None):
    parser = build_parser()
    try:
        execute(parser.parse_args(argv))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
