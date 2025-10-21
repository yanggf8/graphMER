#!/usr/bin/env python3
"""
Quick plot utility to visualize A/B curves from logs CSVs.
Saves PNGs in logs/ by default.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
from typing import List, Dict
import math

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

FIELDS = ["step", "loss", "loss_mlm", "loss_mnm", "val_acc_mlm", "val_acc_mnm"]


def read_csv(path: Path):
    steps, values = [], {k: [] for k in FIELDS if k != "step"}
    if not path.exists():
        return steps, values
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(float(row["step"])) if row.get("step") else math.nan)
            for k in values.keys():
                try:
                    values[k].append(float(row.get(k, "nan")))
                except ValueError:
                    values[k].append(math.nan)
    return steps, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logA", type=str, default=str(Path("logs") / "train_metrics_A.csv"))
    parser.add_argument("--logB", type=str, default=str(Path("logs") / "train_metrics_B.csv"))
    parser.add_argument("--outdir", type=str, default=str(Path("logs")))
    args = parser.parse_args()

    if plt is None:
        print("matplotlib is not available. Please install it to generate plots.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sA, vA = read_csv(Path(args.logA))
    sB, vB = read_csv(Path(args.logB))

    if not sA or not sB:
        print("Missing A or B logs.")
        return

    for metric in ["loss", "loss_mlm", "loss_mnm", "val_acc_mlm", "val_acc_mnm"]:
        plt.figure(figsize=(6,4))
        plt.plot(sA, vA.get(metric, []), label="A: embedding-only")
        plt.plot(sB, vB.get(metric, []), label="B: attn-bias")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        outfile = outdir / f"ablation_{metric}.png"
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
