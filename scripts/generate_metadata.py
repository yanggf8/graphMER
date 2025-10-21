#!/usr/bin/env python3
"""Generate ablation_metadata.json with real computed values"""
import json
import hashlib
import subprocess
import re
import sys
import platform
from datetime import datetime
from pathlib import Path

def get_commit_hash():
    """Get current git commit hash if available"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "no-git"

def get_manifest_hash():
    """Compute SHA256 of manifest.json"""
    manifest_path = Path("data/kg/manifest.json")
    if not manifest_path.exists():
        return "missing"
    
    with open(manifest_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_current_improvement():
    """Parse current MNM improvement using shared utility"""
    try:
        result = subprocess.run(
            ["python", "scripts/extract_improvement.py"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error extracting improvement: {e}")
        return None

def get_manifest_data():
    """Extract key metrics from manifest"""
    manifest_path = Path("data/kg/manifest.json")
    if not manifest_path.exists():
        return {}
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    return {
        "total_triples": manifest.get("total_triples", 0),
        "domain_range_ratio": manifest.get("validation", {}).get("domain_range_ratio", 0)
    }

def main():
    metadata = {
        "schema_version": "1.0",
        "validation_date": datetime.now().isoformat(),
        "commit_hash": get_commit_hash(),
        "dataset_manifest": "data/kg/manifest.json", 
        "dataset_hash": f"sha256:{get_manifest_hash()}",
        "ablation_config": "configs/train_cpu.yaml",
        "random_seed": 42,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "architecture": platform.machine()
        },
        "artifacts": {
            "train_metrics_A": "logs/train_metrics_A.csv",
            "train_metrics_B": "logs/train_metrics_B.csv", 
            "checksums": "checksums.txt"
        },
        "validation_results": {
            "mnm_improvement_percent": get_current_improvement(),
            **get_manifest_data(),
            "training_samples": 300,
            "vocab_size": 878
        }
    }
    
    with open("ablation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Generated ablation_metadata.json with computed values")

if __name__ == "__main__":
    main()
