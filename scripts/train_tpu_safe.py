#!/usr/bin/env python3
"""Safe TPU training with proper metric monitoring"""
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path

def parse_metrics(output):
    """Parse val_acc_mnm and loss_mnm from training output"""
    lines = output.strip().split('\n')
    for line in reversed(lines):
        if 'val acc:' in line:
            # Parse "val acc: mlm=0.636, mnm=0.269"
            mnm_match = re.search(r'mnm=([0-9.]+)', line)
            if mnm_match:
                val_acc_mnm = float(mnm_match.group(1))
                
                # Find corresponding loss line
                for loss_line in reversed(lines):
                    if 'loss=' in loss_line and 'mnm=' in loss_line:
                        loss_match = re.search(r'loss_mnm=([0-9.]+)', loss_line)
                        if loss_match:
                            loss_mnm = float(loss_match.group(1))
                            return {"val_acc_mnm": val_acc_mnm, "loss_mnm": loss_mnm}
                
                return {"val_acc_mnm": val_acc_mnm, "loss_mnm": None}
    
    return {"val_acc_mnm": None, "loss_mnm": None}

def run_tpu_training(config, steps, seed=42, warmup=False):
    """Run TPU training with monitoring"""
    print(f"🚀 {'Warmup' if warmup else 'Full'} TPU training...")
    print(f"   Steps: {steps}, Seed: {seed}")
    
    cmd = [
        "python", "scripts/train.py",
        "--config", config,
        "--steps", str(steps)
    ]
    
    # Set seed via environment
    import os
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return None
    
    metrics = parse_metrics(result.stdout)
    
    # Validate metrics make sense
    val_acc = metrics.get("val_acc_mnm")
    loss = metrics.get("loss_mnm")
    
    print(f"   Final val_acc_mnm: {val_acc:.3f} (higher is better)")
    if loss:
        print(f"   Final loss_mnm: {loss:.3f} (lower is better)")
    
    return {
        "metrics": metrics,
        "output": result.stdout,
        "success": True
    }

def main():
    print("=== Safe TPU Training Pipeline ===\n")
    
    # Step 1: TPU warmup
    print("Step 1: TPU Warmup (300 steps)")
    warmup_result = run_tpu_training("configs/train_tpu.yaml", 300, seed=42, warmup=True)
    
    if not warmup_result:
        print("❌ Warmup failed - aborting")
        return False
    
    warmup_acc = warmup_result["metrics"]["val_acc_mnm"]
    if warmup_acc and warmup_acc > 0.1:  # Reasonable threshold
        print(f"✅ Warmup successful: val_acc_mnm = {warmup_acc:.3f}")
    else:
        print(f"⚠️ Warmup val_acc_mnm = {warmup_acc:.3f} - consider investigating")
        response = input("Continue with full training? (y/N): ")
        if response.lower() != 'y':
            return False
    
    print()
    
    # Step 2: Full TPU training
    print("Step 2: Full TPU Training (5000 steps)")
    full_result = run_tpu_training("configs/train_tpu.yaml", 5000, seed=42)
    
    if not full_result:
        print("❌ Full training failed")
        return False
    
    final_acc = full_result["metrics"]["val_acc_mnm"]
    final_loss = full_result["metrics"]["loss_mnm"]
    
    print(f"✅ Training complete!")
    print(f"   Final val_acc_mnm: {final_acc:.3f}")
    print(f"   Final loss_mnm: {final_loss:.3f}")
    
    # Save training metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    metadata = {
        "training_type": "TPU",
        "timestamp": timestamp,
        "warmup_metrics": warmup_result["metrics"],
        "final_metrics": full_result["metrics"],
        "config": "configs/train_tpu.yaml",
        "seed": 42,
        "total_steps": 5000
    }
    
    with open(f"logs/tpu_training_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📊 Metadata saved: logs/tpu_training_{timestamp}.json")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
