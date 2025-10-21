#!/usr/bin/env python3
"""Test metadata generation and validation"""
import json
import jsonschema
from pathlib import Path

def test_metadata_schema_validation():
    """Test that ablation_metadata.json conforms to schema"""
    metadata_path = Path("ablation_metadata.json")
    schema_path = Path("docs/specs/metadata_schema.json")
    
    if not metadata_path.exists():
        return  # Skip if metadata not generated yet
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    with open(schema_path) as f:
        schema = json.load(f)
    
    # This will raise ValidationError if invalid
    jsonschema.validate(metadata, schema)
    
    # Additional business logic checks
    results = metadata["validation_results"]
    assert results["total_triples"] >= 30000, "Should have 30k+ triples"
    assert results["domain_range_ratio"] >= 0.99, "Should have 99%+ validation quality"
    
    if results["mnm_improvement_percent"] is not None:
        assert results["mnm_improvement_percent"] >= 10, "Should meet 10% threshold"    
    # Check required artifacts exist
    artifacts = metadata["artifacts"]
    for name, path in artifacts.items():
        assert Path(path).exists(), f"Artifact {name} missing at {path}"

if __name__ == "__main__":
    test_metadata_schema_validation()
    print("✅ Metadata validation passed")
