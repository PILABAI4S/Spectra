"""Fresh paired seeds with space for the unchanged estimator seed offsets."""

from spa_former.seeds import generate_seeds, validate_seeds

MAX_TRIAL_SEED = 2**32 - 1 - (200 * 17 + 2)


def generate_paired_seeds(count):
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError("Require1-100000 repetitions")
    result, used = [], set()
    while len(result) < count:
        for seed in generate_seeds(count - len(result)):
            if seed <= MAX_TRIAL_SEED and seed not in used:
                result.append(seed)
                used.add(seed)
    return result


def validate_paired_seeds(seeds):
    seeds = validate_seeds(seeds)
    if any(seed > MAX_TRIAL_SEED for seed in seeds):
        raise ValueError("Manifest seed leaves insufficient uint32 range for the native estimator offsets")
    return seeds


def main():
    import argparse
    from pathlib import Path
    from spa_former.seeds import make_manifest, write_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(args.output, make_manifest(generate_paired_seeds(args.count),
                   "secrets.randbits(32); unique; native estimator offset reserve"))
    print("GENERATED_PRIVATE_MANIFEST " + str(args.output))


if __name__ == "__main__":
    main()
