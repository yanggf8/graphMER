#!/usr/bin/env python3
"""
Automate A/B ablation runs for relation attention bias.
- A: embedding-only HGAT (no attention bias)
- B: attention-bias HGAT

This script does not modify the YAML file; it passes an override to train.py via --rel_bias.
It copies logs/train_metrics.csv after each run to logs/train_metrics_A.csv and logs/train_metrics_B.csv.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/train_cpu.yaml"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=2)
    args = parser.parse_args()

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Ensure KG is built once
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_kg.py")], check=False)
    except FileNotFoundError:
        pass

    # Run A: embedding-only (no relation attention bias)
    run([sys.executable, str(ROOT / "scripts/train.py"), "--config", args.config, "--steps", str(args.steps), "--rel_bias", "false", "--limit", str(args.limit), "--chunk_size", str(args.chunk_size)]) 
    src = logs_dir / "train_metrics.csv"
    dstA = logs_dir / "train_metrics_A.csv"
    if src.exists():
        shutil.copyfile(src, dstA)
        print(f"Saved A logs to {dstA}")
    else:
        print("Warning: train_metrics.csv not found after Run A")

    # Run B: attention-bias
    run([sys.executable, str(ROOT / "scripts/train.py"), "--config", args.config, "--steps", str(args.steps), "--rel_bias", "true", "--limit", str(args.limit), "--chunk_size", str(args.chunk_size)]) 
    dstB = logs_dir / "train_metrics_B.csv"
    if src.exists():
        shutil.copyfile(src, dstB)
        print(f"Saved B logs to {dstB}")
    else:
        print("Warning: train_metrics.csv not found after Run B")

    print("Ablation finished. Use summarize_logs.py to compare.")


if __name__ == "__main__":
    main()
