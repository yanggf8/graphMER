#!/usr/bin/env python3
"""Comprehensive production metadata validation."""

import json
import sys
from pathlib import Path

def validate_production_metadata():
    """Validate production metadata meets all requirements."""
    
    try:
        with open("ablation_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("❌ No ablation_metadata.json found")
        return False
    
    run_name = metadata.get("run_name", "")
    
    if not run_name.startswith("production"):
        print(f"✅ Non-production run '{run_name}', validation skipped")
        return True
    
    print(f"🔍 Validating production run: {run_name}")
    
    # Schema version check
    schema_version = metadata.get("schema_version")
    if schema_version != "1.1":
        print(f"❌ Production runs must use schema 1.1, got: {schema_version}")
        return False
    
    # Artifacts and checksums check
    artifacts = metadata.get("artifacts", {})
    checksums = metadata.get("artifact_checksums", {})
    
    if not artifacts:
        print("❌ Production run missing artifacts")
        return False
    
    for artifact_type, path in artifacts.items():
        # Check checksum exists and is valid
        checksum = checksums.get(artifact_type)
        if not checksum:
            print(f"❌ Missing checksum for {artifact_type}")
            return False
        
        if not checksum.startswith("sha256:") or len(checksum) < 70:
            print(f"❌ Invalid checksum for {artifact_type}: {checksum}")
            return False
        
        # Check file exists and meets size thresholds
        artifact_path = Path(path)
        if not artifact_path.exists():
            print(f"❌ Artifact file missing: {path}")
            return False
        
        size = artifact_path.stat().st_size
        if path.endswith(".csv") and size < 200:
            print(f"❌ Production CSV too small: {path} ({size} bytes)")
            return False
        elif path.endswith(".pt") and size < 1000:
            print(f"❌ Production checkpoint too small: {path} ({size} bytes)")
            return False
    
    # Check required fields
    required_fields = ["run_id", "validation_date", "dataset_hash", "validation_results"]
    for field in required_fields:
        if field not in metadata:
            print(f"❌ Missing required field: {field}")
            return False
    
    print(f"✅ Production run {run_name} passes all validation checks")
    print(f"  Schema: {schema_version}")
    print(f"  Artifacts: {len(artifacts)} with valid checksums")
    print(f"  Run ID: {metadata.get('run_id', '')[:8]}...")
    
    return True

if __name__ == "__main__":
    success = validate_production_metadata()
    sys.exit(0 if success else 1)
