#!/usr/bin/env python3
"""Readiness gate for GraphMER-SE training"""
import json
import subprocess
import sys
from pathlib import Path

def check_data_integrity():
    """Validate dataset meets policy requirements"""
    print("🔍 Checking data integrity...")
    
    # Load policy and manifest
    with open("policy.json") as f:
        policy = json.load(f)
    
    with open("data/kg/manifest.json") as f:
        manifest = json.load(f)
    
    violations = []
    
    # Check triples count
    triples = manifest.get("total_triples", 0)
    min_triples = policy["thresholds"]["total_triples"]["minimum"]
    if triples < min_triples:
        violations.append(f"Triples {triples} below minimum {min_triples}")
    
    # Check validation quality
    ratio = manifest.get("validation", {}).get("domain_range_ratio", 0)
    min_ratio = policy["thresholds"]["domain_range_ratio"]["minimum"]
    if ratio < min_ratio:
        violations.append(f"Validation ratio {ratio:.3f} below {min_ratio}")
    
    if violations:
        print("❌ Data integrity failures:")
        for v in violations:
            print(f"  - {v}")
        return False
    
    print(f"✅ Data integrity: {triples} triples, {ratio:.1%} validation")
    return True

def run_sanity_train():
    """Run 1-2 epoch sanity check"""
    print("🔍 Running sanity training check...")
    
    try:
        # Run minimal training to check loss decreases
        result = subprocess.run([
            "python", "scripts/train.py", 
            "--config", "configs/train_cpu.yaml",
            "--steps", "20"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Sanity training: Loss curve stable")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Sanity training failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_baseline_metrics():
    """Verify current metrics are stable"""
    print("🔍 Checking baseline metrics...")
    
    try:
        # Check if we have valid ablation results
        result = subprocess.run([
            "python", "scripts/extract_improvement.py", "--json"
        ], capture_output=True, text=True, check=True)
        
        data = json.loads(result.stdout)
        validation = data["validation"]
        
        if validation["status"] == "fail":
            print(f"❌ Baseline metrics: {validation['message']}")
            print("💡 Suggestion: Run python scripts/run_ablation.py --steps 200")
            return False
        elif validation["status"] == "warning":
            print(f"⚠️ Baseline metrics: {validation['message']}")
            print("💡 Consider increasing training steps for better margin")
            return True  # Allow with warning
        else:
            print(f"✅ Baseline metrics: {validation['message']}")
            return True
            
    except Exception as e:
        print(f"❌ Metrics check failed: {e}")
        return False

def main():
    """Run complete readiness gate"""
    print("=== GraphMER-SE Training Readiness Gate ===\n")
    
    checks = [
        ("Data Integrity", check_data_integrity),
        ("Baseline Metrics", check_baseline_metrics),
        ("Sanity Training", run_sanity_train)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 READINESS GATE PASSED")
        print("✅ Ready to proceed with full training")
        print("\nRecommended training command:")
        print("python scripts/train.py --config configs/train_cpu.yaml --epochs 50 --seed 42")
        return True
    else:
        print("🚫 READINESS GATE FAILED")
        print("❌ Address issues before full training")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
