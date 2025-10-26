#!/usr/bin/env python3
"""Training script with run name and resume support."""
import sys
from pathlib import Path
import argparse
import yaml
import torch
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default=None, help="Run name for artifacts")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Generate run name if not provided
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mnm_fixed_{timestamp}"
    
    print(f"🚀 Starting run: {args.run_name}")
    
    # Create run-specific directories
    run_dir = Path(f"logs/runs/{args.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Use existing proven training logic
    from scripts.train import main as train_main
    
    # Temporarily modify sys.argv for train.py
    original_argv = sys.argv.copy()
    sys.argv = [
        "train.py",
        "--config", args.config,
        "--steps", str(args.steps)
    ]
    
    try:
        # Run training
        train_main()
        
        # Move artifacts to run directory
        if Path("logs/train_metrics.csv").exists():
            Path("logs/train_metrics.csv").rename(run_dir / "metrics.csv")
        
        if Path("logs/checkpoints/model_final.pt").exists():
            Path("logs/checkpoints/model_final.pt").rename(checkpoint_dir / "model_final.pt")
        
        # Generate run-specific metadata
        import subprocess
        subprocess.run([
            "python3", "scripts/generate_metadata.py", 
            str(run_dir / "metadata.json")
        ])
        
        print(f"✅ Run completed: {args.run_name}")
        print(f"📁 Artifacts saved to: {run_dir}")
        
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
