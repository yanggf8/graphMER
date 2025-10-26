#!/usr/bin/env python3
"""Prepare production release with artifact manifest."""

import json
import hashlib
from pathlib import Path
from datetime import datetime

def create_release_manifest():
    """Create signed artifact manifest for release."""
    
    # Load production metadata
    with open("ablation_metadata.json") as f:
        metadata = json.load(f)
    
    run_name = metadata.get("run_name", "unknown")
    
    # Create release manifest
    manifest = {
        "release_version": "v1.1.0-prod-ready",
        "release_date": datetime.now().isoformat(),
        "run_name": run_name,
        "schema_version": metadata.get("schema_version"),
        "artifacts": {},
        "verification": {
            "total_triples": metadata.get("validation_results", {}).get("total_triples"),
            "mnm_accuracy": metadata.get("training_results", {}).get("final_mnm_accuracy"),
            "mlm_accuracy": metadata.get("training_results", {}).get("final_mlm_accuracy")
        }
    }
    
    # Add artifact checksums
    artifacts = metadata.get("artifacts", {})
    checksums = metadata.get("artifact_checksums", {})
    
    for artifact_type, path in artifacts.items():
        if Path(path).exists():
            manifest["artifacts"][artifact_type] = {
                "path": path,
                "checksum": checksums.get(artifact_type),
                "size_bytes": Path(path).stat().st_size
            }
    
    # Create manifest checksum
    manifest_content = json.dumps(manifest, sort_keys=True, indent=2)
    manifest_hash = hashlib.sha256(manifest_content.encode()).hexdigest()
    manifest["manifest_checksum"] = f"sha256:{manifest_hash}"
    
    # Write release manifest
    with open("release_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Created release_manifest.json")
    print(f"  Version: {manifest['release_version']}")
    print(f"  Run: {run_name}")
    print(f"  Artifacts: {len(manifest['artifacts'])}")
    print(f"  Manifest checksum: {manifest_hash[:16]}...")
    
    return manifest

if __name__ == "__main__":
    create_release_manifest()
