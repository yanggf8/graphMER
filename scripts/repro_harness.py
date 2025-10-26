#!/usr/bin/env python3
"""One-command reproducibility harness for GraphMER-SE"""
import subprocess
import json
import sys
from pathlib import Path

def run_step(name, cmd, check=True, timeout=600, shell=False):
    """Run a step with real-time logging and timeout safety.

    Args:
        name: Friendly step name.
        cmd: List of args (recommended) or string if shell=True.
        check: If True, non-zero exit triggers failure.
        timeout: Seconds before we abort the step.
        shell: Use shell execution when necessary.
    """
    print(f"🔄 {name}...")
    try:
        # Stream output live for visibility
        proc = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        lines = []
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                lines.append(line)
                print(line.rstrip())
        except Exception:
            # Fallback: wait with timeout
            proc.wait(timeout=timeout)
        finally:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"⏱️ {name} timed out after {timeout}s. Aborting this step.")
                return None
        if check and proc.returncode != 0:
            print(f"❌ {name} failed with code {proc.returncode}")
            tail = ''.join(lines[-50:])
            if tail:
                print("Last output:\n" + tail)
            return None
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name} crashed: {e}")
        return None

def validate_policy():
    """Validate against policy.json thresholds"""
    with open("policy.json") as f:
        policy = json.load(f)
    
    # Check metrics
    result = subprocess.run(
        ["python", "scripts/extract_improvement.py", "--json"],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    
    # Check manifest
    with open("data/kg/manifest.json") as f:
        manifest = json.load(f)
    
    violations = []
    thresholds = policy["thresholds"]
    
    # Validate improvement metrics
    val_acc = data["metrics"].get("val_acc_improvement", 0)
    if val_acc < thresholds["mnm_val_acc_improvement"]["fail_below"]:
        violations.append(f"Val_acc {val_acc}% below {thresholds['mnm_val_acc_improvement']['fail_below']}%")
    
    # Validate manifest metrics
    triples = manifest.get("total_triples", 0)
    if triples < thresholds["total_triples"]["minimum"]:
        violations.append(f"Triples {triples} below {thresholds['total_triples']['minimum']}")
    
    ratio = manifest.get("validation", {}).get("domain_range_ratio", 0)
    if ratio < thresholds["domain_range_ratio"]["minimum"]:
        violations.append(f"Validation ratio {ratio:.3f} below {thresholds['domain_range_ratio']['minimum']}")
    
    return violations

def main():
    print("=== GraphMER-SE Reproducibility Harness ===\n")
    
    steps = [
        ("Rebuild KG", "python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples"),
        ("Run mini ablation", "python scripts/run_ablation.py --config configs/train_cpu.yaml --steps 100"),
        ("Generate metadata", "python scripts/generate_metadata.py"),
        ("Update checksums", "sha256sum logs/train_metrics_A.csv logs/train_metrics_B.csv ablation_metadata.json > checksums.txt"),
        ("Validate schema", "python tests/test_metadata.py"),
        ("Run verification", "python scripts/verify_production_claims.py")
    ]
    
    for name, cmd in steps:
        if not run_step(name, cmd, check=True, timeout=1200, shell=True):
            sys.exit(1)
    
    # Policy validation
    print("🔄 Validating policy compliance...")
    violations = validate_policy()
    if violations:
        print("❌ Policy violations:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("✅ Policy compliance validated")
    
    print("\n🎉 REPRODUCIBILITY HARNESS COMPLETE")
    print("Status: All steps passed, policy compliant, artifacts verified")

if __name__ == "__main__":
    main()
