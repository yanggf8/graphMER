#!/usr/bin/env python3
"""Production training wrapper with safety features"""
import argparse
import subprocess
import json
import os
import time
from datetime import datetime
from pathlib import Path

def run_training_epoch(config, steps_per_epoch, seed, log_dir, rel_bias="config"):
    """Run one training epoch and return metrics"""
    cmd = [
        "python", "scripts/train.py",
        "--config", config,
        "--steps", str(steps_per_epoch),
        "--rel_bias", rel_bias
    ]
    
    # Set seed via environment (if train.py supports it)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Training failed: {result.stderr}")
        return None
    
    # Parse final metrics from output
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if 'val acc:' in line:
            # Extract val_acc_mnm from line like "val acc: mlm=0.857, mnm=0.250"
            parts = line.split('mnm=')
            if len(parts) > 1:
                val_acc_mnm = float(parts[1].strip())
                return {"val_acc_mnm": val_acc_mnm, "output": result.stdout}
    
    return {"val_acc_mnm": 0.0, "output": result.stdout}

def main():
    parser = argparse.ArgumentParser(description="Production training with safety features")
    parser.add_argument("--config", default="configs/train_cpu.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--warmup-only", action="store_true", help="Run 5-epoch warmup only")
    
    args = parser.parse_args()
    
    # Setup logging directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.log_dir = f"logs/prod_run_{timestamp}"
    
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"=== Production Training ===")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs if not args.warmup_only else 5}")
    print(f"Seed: {args.seed}")
    print(f"Log dir: {args.log_dir}")
    print()
    
    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    epochs_to_run = 5 if args.warmup_only else args.epochs
    
    training_log = []
    
    for epoch in range(1, epochs_to_run + 1):
        print(f"Epoch {epoch}/{epochs_to_run}")
        
        # Run training epoch
        metrics = run_training_epoch(
            args.config, 
            args.steps_per_epoch, 
            args.seed + epoch,  # Vary seed slightly per epoch
            args.log_dir
        )
        
        if metrics is None:
            print("❌ Training failed")
            return False
        
        val_acc = metrics["val_acc_mnm"]
        training_log.append({
            "epoch": epoch,
            "val_acc_mnm": val_acc,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"  Val acc (MNM): {val_acc:.3f}")
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"  ✅ New best: {best_val_acc:.3f}")
            
            # Save best checkpoint info
            with open(f"{args.log_dir}/best_checkpoint.json", "w") as f:
                json.dump({
                    "epoch": epoch,
                    "val_acc_mnm": best_val_acc,
                    "seed": args.seed + epoch
                }, f, indent=2)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping after {epoch} epochs")
                break
        
        print()
    
    # Save training log
    with open(f"{args.log_dir}/training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"🎉 Training complete!")
    print(f"Best val_acc_mnm: {best_val_acc:.3f}")
    print(f"Logs saved to: {args.log_dir}")
    
    if args.warmup_only:
        print("\n💡 Warmup successful. Run without --warmup-only for full training.")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
