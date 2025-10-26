#!/usr/bin/env python3
"""Monitor production training run."""
import time
import re
from pathlib import Path

def monitor_run(log_path="logs/production_v1.log", run_name="production_v1"):
    """Monitor training progress and provide updates."""
    
    print(f"🔍 Monitoring production run: {run_name}")
    print("=" * 50)
    
    last_position = 0
    step_pattern = re.compile(r'step (\d+): total_loss=([\d.]+), mlm_loss=([\d.]+), mnm_loss=([\d.]+)')
    acc_pattern = re.compile(r'val_acc: mlm_accuracy=([\d.]+), mnm_accuracy=([\d.]+)')
    
    latest_metrics = {}
    
    while True:
        try:
            if Path(log_path).exists():
                with open(log_path, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                
                for line in new_lines:
                    line = line.strip()
                    
                    # Parse step metrics
                    step_match = step_pattern.search(line)
                    if step_match:
                        step, total_loss, mlm_loss, mnm_loss = step_match.groups()
                        latest_metrics.update({
                            'step': int(step),
                            'total_loss': float(total_loss),
                            'mlm_loss': float(mlm_loss),
                            'mnm_loss': float(mnm_loss)
                        })
                    
                    # Parse accuracy metrics
                    acc_match = acc_pattern.search(line)
                    if acc_match:
                        mlm_acc, mnm_acc = acc_match.groups()
                        latest_metrics.update({
                            'mlm_acc': float(mlm_acc),
                            'mnm_acc': float(mnm_acc)
                        })
                        
                        # Print progress update
                        if 'step' in latest_metrics:
                            step = latest_metrics['step']
                            progress = (step / 5000) * 100
                            print(f"Step {step:4d} ({progress:5.1f}%): "
                                  f"Loss={latest_metrics['total_loss']:.4f}, "
                                  f"MLM={latest_metrics['mlm_acc']:.3f}, "
                                  f"MNM={latest_metrics['mnm_acc']:.3f}")
                            
                            # Check for MNM breakthrough
                            if latest_metrics['mnm_acc'] > 0.5:
                                print(f"🎉 MNM breakthrough! Accuracy: {latest_metrics['mnm_acc']:.1%}")
                    
                    # Check for completion
                    if "Training completed" in line or "Smoke training finished" in line:
                        print("\n✅ Production run completed!")
                        return latest_metrics
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⏸️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(10)
    
    return latest_metrics

if __name__ == "__main__":
    final_metrics = monitor_run()
    if final_metrics:
        print(f"\n📊 Final metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
