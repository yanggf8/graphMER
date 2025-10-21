#!/usr/bin/env python3
"""Shared utility to extract MNM improvement percentage with multi-metric validation"""
import subprocess
import re
import sys
import json

def get_mnm_metrics():
    """Extract both MNM val_acc improvement and loss change"""
    try:
        result = subprocess.run(
            ["python", "scripts/summarize_logs.py", "--steps", "100"],
            capture_output=True, text=True, check=True
        )
        
        metrics = {}
        
        # Extract val_acc improvement
        for line in result.stdout.split('\n'):
            if "Relative improvement (MNM val_acc, higher is better)" in line:
                match = re.search(r":\s*([0-9.]+)%", line)
                if match:
                    metrics["val_acc_improvement"] = float(match.group(1))
            elif "Relative improvement (MNM loss, lower is better)" in line:
                match = re.search(r":\s*(-?[0-9.]+)%", line)
                if match:
                    metrics["loss_improvement"] = float(match.group(1))
        
        return metrics
    except:
        return {}

def validate_metrics(metrics, warning_threshold=12.0, fail_threshold=10.0):
    """Validate metrics with warning band"""
    if not metrics:
        return {"status": "error", "message": "No metrics found"}
    
    val_acc = metrics.get("val_acc_improvement", 0)
    loss_change = metrics.get("loss_improvement", 0)
    
    # Multi-metric validation
    if val_acc < fail_threshold:
        return {"status": "fail", "message": f"Val_acc {val_acc}% below {fail_threshold}% threshold"}
    elif loss_change > 0:  # Loss should improve (negative) or stay same
        return {"status": "fail", "message": f"Loss degraded by {loss_change}%"}
    elif val_acc < warning_threshold:
        return {"status": "warning", "message": f"Val_acc {val_acc}% in warning band ({fail_threshold}-{warning_threshold}%)"}
    else:
        return {"status": "pass", "message": f"Val_acc {val_acc}% passes with margin"}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        metrics = get_mnm_metrics()
        validation = validate_metrics(metrics)
        print(json.dumps({"metrics": metrics, "validation": validation}))
    else:
        # Backward compatibility - return just val_acc improvement
        metrics = get_mnm_metrics()
        val_acc = metrics.get("val_acc_improvement")
        if val_acc is None:
            print("ERROR: Could not extract improvement. Run: python scripts/run_ablation.py --steps 200", file=sys.stderr)
            sys.exit(1)
        print(f"{val_acc}")
