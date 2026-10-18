"""Generate fresh seeds and retain an exact, replayable manifest."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets

MAX_SEED = 2**32 - 1


def validate_seeds(seeds):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")
    if any(type(seed) is not int or not 0 <= seed <= MAX_SEED for seed in seeds):
        raise ValueError("Seeds must be integers in [0, 2**32 - 1].")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique; duplicate seeds would overwrite runs.")
    return seeds


def generate_seeds(count=20):
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError("Seed count must be an integer in [1, 100000].")
    seeds, used = [], set()
    while len(seeds) < count:
        seed = secrets.randbits(32)
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    return seeds


def make_manifest(seeds, policy):
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed_generation": policy, "seeds": validate_seeds(seeds)}


def write_manifest(path, manifest):
    validate_seeds(manifest["seeds"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return validate_seeds(manifest["seeds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = make_manifest(generate_seeds(args.count), "secrets.randbits(32); unique")
        write_manifest(args.output, manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
