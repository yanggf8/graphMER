#!/usr/bin/env python3
"""Keep only the latest N checkpoints to save disk space."""
import os
from pathlib import Path
import argparse

def cleanup_checkpoints(keep_latest=3):
    """Keep only the latest N checkpoints, delete older ones."""
    checkpoint_dir = Path("logs/checkpoints")
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files
    checkpoints = list(checkpoint_dir.glob("model_v2_step*.pt"))
    if len(checkpoints) <= keep_latest:
        return
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Delete older checkpoints
    for old_checkpoint in checkpoints[keep_latest:]:
        old_checkpoint.unlink()
        print(f"Deleted: {old_checkpoint.name}")
    
    print(f"Kept latest {min(len(checkpoints), keep_latest)} checkpoints")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep")
    args = parser.parse_args()
    cleanup_checkpoints(args.keep)
