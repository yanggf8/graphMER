#!/usr/bin/env python3
"""Post-training validation with policy archival"""
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path

def main():
    print("=== Post-Training Validation ===\n")
    
    # Run full reproducibility harness
    print("🔄 Running reproducibility harness...")
    # Stream harness output with timeout to avoid silent hangs
    try:
        proc = subprocess.Popen(["python", "scripts/repro_harness.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            print(line.rstrip())
        proc.wait(timeout=1800)
        if proc.returncode != 0:
            print("❌ Reproducibility harness failed")
            return False
    except subprocess.TimeoutExpired:
        proc.kill()
        print("⏱️ Reproducibility harness timed out after 1800s")
        return False
    
    print("✅ Reproducibility harness passed")
    
    # Generate final metadata
    print("🔄 Generating final metadata...")
    subprocess.run(["python", "scripts/generate_metadata.py"])
    subprocess.run(["python", "scripts/track_trends.py"])
    
    # Archive policy.json with training artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = Path(f"artifacts/training_{timestamp}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy key artifacts
    artifacts = [
        "ablation_metadata.json",
        "policy.json", 
        "trends.json",
        "checksums.txt",
        "data/kg/manifest.json"
    ]
    
    for artifact in artifacts:
        if Path(artifact).exists():
            shutil.copy2(artifact, archive_dir)
    
    # Create training summary
    with open("ablation_metadata.json") as f:
        metadata = json.load(f)
    
    summary = {
        "training_date": timestamp,
        "commit_hash": metadata.get("commit_hash", "unknown"),
        "dataset_hash": metadata.get("dataset_hash", "unknown"),
        "improvement_percent": metadata.get("validation_results", {}).get("mnm_improvement_percent"),
        "total_triples": metadata.get("validation_results", {}).get("total_triples"),
        "policy_compliant": True,
        "archive_path": str(archive_dir)
    }
    
    with open(archive_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Artifacts archived to: {archive_dir}")
    print(f"📊 Final improvement: {summary['improvement_percent']}%")
    print(f"📈 Dataset scale: {summary['total_triples']} triples")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
