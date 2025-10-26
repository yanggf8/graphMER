#!/usr/bin/env python3
"""CI check to prevent production metadata regression."""

import json
import sys

def check_production_metadata_regression():
    """Ensure production metadata doesn't regress."""
    
    try:
        with open("ablation_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("✅ No metadata file, CI check passed")
        return True
    
    run_name = metadata.get("run_name", "")
    
    # Only check production runs
    if not run_name.startswith("production"):
        print(f"✅ Non-production run '{run_name}', CI check passed")
        return True
    
    # Check for regression to schema 1.0
    schema_version = metadata.get("schema_version")
    if schema_version == "1.0":
        print(f"❌ CI FAIL: Production run {run_name} regressed to schema 1.0")
        return False
    
    # Check for missing checksums
    artifacts = metadata.get("artifacts", {})
    checksums = metadata.get("artifact_checksums", {})
    
    for artifact_type in artifacts:
        if not checksums.get(artifact_type):
            print(f"❌ CI FAIL: Production run {run_name} missing checksum for {artifact_type}")
            return False
    
    print(f"✅ CI check passed for production run {run_name}")
    return True

if __name__ == "__main__":
    success = check_production_metadata_regression()
    sys.exit(0 if success else 1)
