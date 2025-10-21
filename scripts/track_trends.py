#!/usr/bin/env python3
"""Simple trend tracking for GraphMER-SE metrics"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

def update_trends():
    """Append current metrics to trends.json"""
    trends_file = Path("trends.json")
    
    # Get current metrics
    result = subprocess.run(
        ["python", "scripts/extract_improvement.py", "--json"],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    
    # Get manifest data
    with open("data/kg/manifest.json") as f:
        manifest = json.load(f)
    
    # Create trend entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "CPU",  # Default, can be overridden
        "val_acc_improvement": data["metrics"].get("val_acc_improvement"),
        "total_triples": manifest.get("total_triples"),
        "domain_range_ratio": manifest.get("validation", {}).get("domain_range_ratio"),
        "build_duration": manifest.get("build_duration_sec")
    }
    
    # Load existing trends
    if trends_file.exists():
        with open(trends_file) as f:
            trends = json.load(f)
    else:
        trends = []
    
    # Append and keep last 50 entries
    trends.append(entry)
    trends = trends[-50:]
    
    # Save trends
    with open(trends_file, "w") as f:
        json.dump(trends, f, indent=2)
    
    print(f"✅ Updated trends.json with {len(trends)} entries")

if __name__ == "__main__":
    update_trends()
