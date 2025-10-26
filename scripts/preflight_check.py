#!/usr/bin/env python3
"""Pre-flight check for production completion."""

import sys
from pathlib import Path

def preflight_check(run_name):
    """Check if production run is ready for completion workflow."""
    
    print(f"🔍 Pre-flight check for: {run_name}")
    
    # Check if run directory exists
    run_dir = Path(f"logs/runs/{run_name}")
    if not run_dir.exists():
        print(f"❌ Run directory missing: {run_dir}")
        print("   Training may still be in progress or artifacts not moved yet")
        return False
    
    # Check for expected artifacts
    expected_files = [
        run_dir / "metrics.csv",
        run_dir / "checkpoints" / "model_final.pt"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Missing expected artifacts:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("   Training may still be in progress")
        return False
    
    # Check file sizes (ensure not placeholders)
    metrics_file = run_dir / "metrics.csv"
    checkpoint_file = run_dir / "checkpoints" / "model_final.pt"
    
    metrics_size = metrics_file.stat().st_size
    checkpoint_size = checkpoint_file.stat().st_size
    
    if metrics_size < 1000:  # Should have many training steps
        print(f"⚠️ Metrics file suspiciously small: {metrics_size} bytes")
        print("   Training may not be complete")
        return False
    
    if checkpoint_size < 10000:  # Model should be substantial
        print(f"⚠️ Checkpoint file suspiciously small: {checkpoint_size} bytes")
        print("   Training may not be complete")
        return False
    
    # Check if training process is still running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', run_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"⚠️ Training process still running for {run_name}")
            print("   Wait for training to complete before running production workflow")
            return False
    except:
        pass  # pgrep not available or other error
    
    print(f"✅ Pre-flight check passed for {run_name}")
    print(f"   Metrics: {metrics_size:,} bytes")
    print(f"   Checkpoint: {checkpoint_size:,} bytes")
    print(f"   Ready for production completion workflow")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Production run name to check")
    args = parser.parse_args()
    
    success = preflight_check(args.run_name)
    
    if success:
        print(f"\n🚀 Ready to run:")
        print(f"   make production-complete RUN_NAME={args.run_name}")
    else:
        print(f"\n⏳ Wait for training completion, then retry:")
        print(f"   python3 scripts/preflight_check.py {args.run_name}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
