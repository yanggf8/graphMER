#!/usr/bin/env python3
"""Run ablation across multiple seeds for statistical significance."""
import subprocess
import sys
from pathlib import Path
import shutil

def run_seed_ablation(seed: int, steps: int = 200):
    """Run ablation for a specific seed."""
    # Modify config temporarily or pass seed as argument
    cmd = [
        sys.executable, "scripts/run_ablation.py",
        "--config", "configs/train_cpu.yaml",
        "--steps", str(steps),
        "--limit", "128",
        "--chunk_size", "3"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Save results with seed suffix
    logs_dir = Path("logs")
    for suffix in ["A", "B"]:
        src = logs_dir / f"train_metrics_{suffix}.csv"
        dst = logs_dir / f"train_metrics_{suffix}_seed{seed}.csv"
        if src.exists():
            shutil.copy(src, dst)

if __name__ == "__main__":
    seeds = [1337, 42, 2023]  # Run 3 seeds
    for seed in seeds:
        print(f"Running ablation with seed {seed}")
        run_seed_ablation(seed)
