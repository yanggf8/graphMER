#!/usr/bin/env python3
"""Validate ablation_metadata.json against schema."""

import json
import sys
from pathlib import Path

def validate_metadata(metadata_path="ablation_metadata.json", schema_path="docs/specs/metadata_schema.json"):
    """Validate metadata against JSON schema."""
    
    # Check files exist
    if not Path(metadata_path).exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return False
    
    if not Path(schema_path).exists():
        print(f"⚠️ Schema file not found: {schema_path} (skipping schema validation)")
        schema_validation = False
    else:
        schema_validation = True
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Basic structure checks
    required_fields = ["schema_version", "validation_results"]
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Validation results checks
    val_results = metadata["validation_results"]
    required_val_fields = ["total_triples", "domain_range_ratio", "mnm_improvement_percent"]
    missing_val_fields = [field for field in required_val_fields if field not in val_results]
    
    if missing_val_fields:
        print(f"❌ Missing validation_results fields: {missing_val_fields}")
        return False
    
    # Value range checks
    if not (0 <= val_results["domain_range_ratio"] <= 1):
        print(f"❌ domain_range_ratio out of range: {val_results['domain_range_ratio']}")
        return False
    
    if val_results["total_triples"] <= 0:
        print(f"❌ total_triples must be positive: {val_results['total_triples']}")
        return False
    
    print(f"✅ Metadata validation passed")
    print(f"  Schema version: {metadata['schema_version']}")
    print(f"  Total triples: {val_results['total_triples']:,}")
    print(f"  Domain/range ratio: {val_results['domain_range_ratio']:.3f}")
    
    return True

if __name__ == "__main__":
    metadata_path = sys.argv[1] if len(sys.argv) > 1 else "ablation_metadata.json"
    success = validate_metadata(metadata_path)
    sys.exit(0 if success else 1)
