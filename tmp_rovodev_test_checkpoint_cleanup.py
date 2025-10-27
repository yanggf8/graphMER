#!/usr/bin/env python3
"""Test checkpoint cleanup logic to verify Amazon Q's implementation."""

from pathlib import Path
from datetime import datetime
import time

def test_checkpoint_cleanup():
    """Simulate the checkpoint cleanup logic."""
    
    print("=" * 60)
    print("Testing Checkpoint Cleanup Logic")
    print("=" * 60)
    print()
    
    # Create test directory
    test_dir = Path("logs/test_checkpoints")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Test directory: {test_dir}")
    print()
    
    # Create mock checkpoint files
    print("Creating 5 mock checkpoint files...")
    for i in range(1, 6):
        checkpoint_path = test_dir / f"model_v2_step{i*1000}_test_s42.pt"
        checkpoint_path.write_text(f"Mock checkpoint {i}")
        time.sleep(0.1)  # Ensure different timestamps
        print(f"  Created: {checkpoint_path.name}")
    
    print()
    print("Checkpoints before cleanup:")
    checkpoints = sorted(test_dir.glob("model_v2_step*.pt"), key=lambda x: x.stat().st_mtime)
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp.name} (modified: {datetime.fromtimestamp(cp.stat().st_mtime).strftime('%H:%M:%S')})")
    
    print()
    print("Applying cleanup logic (keep only latest 2)...")
    
    # This is the exact logic from train_v2.py lines 355-358
    checkpoints = sorted(test_dir.glob("model_v2_step*.pt"), key=lambda x: x.stat().st_mtime)
    deleted_count = 0
    for old_cp in checkpoints[:-2]:
        old_cp.unlink()
        print(f"  ✓ Deleted: {old_cp.name}")
        deleted_count += 1
    
    print()
    print("Checkpoints after cleanup:")
    remaining = sorted(test_dir.glob("model_v2_step*.pt"), key=lambda x: x.stat().st_mtime)
    for i, cp in enumerate(remaining, 1):
        print(f"  {i}. {cp.name} (modified: {datetime.fromtimestamp(cp.stat().st_mtime).strftime('%H:%M:%S')})")
    
    print()
    print("=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"✓ Initial checkpoints: 5")
    print(f"✓ Deleted checkpoints: {deleted_count}")
    print(f"✓ Remaining checkpoints: {len(remaining)}")
    print(f"✓ Expected remaining: 2")
    
    if len(remaining) == 2:
        print()
        print("✅ SUCCESS: Cleanup logic works correctly!")
        print("   Kept the 2 most recent checkpoints as expected.")
        
        # Verify they are the latest ones
        names = [cp.name for cp in remaining]
        if "model_v2_step4000_test_s42.pt" in names and "model_v2_step5000_test_s42.pt" in names:
            print("   ✓ Correct checkpoints retained (step 4000 and 5000)")
        else:
            print(f"   ⚠️  Unexpected checkpoints retained: {names}")
    else:
        print()
        print(f"❌ FAILED: Expected 2 remaining, got {len(remaining)}")
    
    # Cleanup test directory
    print()
    print("Cleaning up test directory...")
    for cp in test_dir.glob("*.pt"):
        cp.unlink()
    test_dir.rmdir()
    print("✓ Test directory cleaned up")
    print()

if __name__ == "__main__":
    test_checkpoint_cleanup()
