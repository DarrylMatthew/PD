"""Run baseline.py for every (data root, train dataset, seed) and log to results/baseline.csv.

Resumable: combinations already present in the CSV are skipped.
Usage: python scripts/run_baselines.py --seeds 1 2 3
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

DATASETS = ["newhandpd", "handpd", "parkd"]
ROOTS = ["data", "data_norm"]
RESULTS = Path("results/baseline.csv")


def done_runs():
    if not RESULTS.exists():
        return set()
    with open(RESULTS, newline="") as f:
        return {(r["tag"], r["train_dataset"], int(r["seed"]), int(r["epochs"]))
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    args = ap.parse_args()

    RESULTS.parent.mkdir(exist_ok=True)
    for seed in args.seeds:
        for root in args.roots:
            for train_ds in DATASETS:
                if (Path(root).name, train_ds, seed, args.epochs) in done_runs():
                    print(f"skip {root} {train_ds} seed {seed}")
                    continue
                others = [d for d in DATASETS if d != train_ds]
                print(f"=== {root} | train {train_ds} | seed {seed} ===", flush=True)
                subprocess.run(
                    [sys.executable, "baseline.py", "--data-root", root, "--train-dataset", train_ds,
                     "--test-datasets", *others, "--seed", str(seed), "--epochs", str(args.epochs),
                     "--results-csv", str(RESULTS)],
                    check=True)


if __name__ == "__main__":
    main()
