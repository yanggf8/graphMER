#!/usr/bin/env python3
"""TPU dry-run for baseline establishment"""
import time
import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--config", type=str, default="configs/train_tpu.yaml")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Import after args to avoid TPU init overhead
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    
    from scripts.train_fixed import main as train_main
    import sys
    
    # Override sys.argv for train_fixed
    sys.argv = ["train_fixed.py", "--config", args.config, "--steps", str(args.steps), "--limit", "512", "--chunk_size", "8"]
    
    try:
        train_main()
        duration = time.time() - start_time
        
        # Log TPU metrics
        metrics = {
            "duration_sec": duration,
            "steps": args.steps,
            "tokens_per_sec": (args.steps * 512) / duration,  # rough estimate
            "status": "success",
            "timestamp": time.time()
        }
        
        with open("logs/tpu_baseline.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"TPU dry-run complete: {duration:.1f}s, ~{metrics['tokens_per_sec']:.0f} tokens/sec")
        
    except Exception as e:
        print(f"TPU dry-run failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
