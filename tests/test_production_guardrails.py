"""Test production guardrails and safety checks."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_production_run_no_placeholders():
    """Test that production runs never reference placeholder paths."""
    
    # Load current metadata
    try:
        with open("ablation_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("⚠️ No metadata file found, skipping production guardrail test")
        return
    
    run_name = metadata.get("run_name", "")
    artifacts = metadata.get("artifacts", {})
    
    # Check if this is a production run
    if run_name.startswith("production"):
        for artifact_type, path in artifacts.items():
            # Production runs should not use test artifact paths
            assert "test_artifacts" not in path, f"Production run {run_name} uses test artifact: {path}"
            
            # Check file sizes (placeholders are typically very small)
            artifact_path = Path(path)
            if artifact_path.exists():
                size = artifact_path.stat().st_size
                if path.endswith(".csv"):
                    assert size > 200, f"Production CSV too small (likely placeholder): {path} ({size} bytes)"
                elif path.endswith(".pt"):
                    assert size > 1000, f"Production checkpoint too small (likely placeholder): {path} ({size} bytes)"
        
        print(f"✅ Production run {run_name} passes guardrail checks")
    else:
        print(f"✅ Non-production run {run_name}, guardrails skipped")

def test_schema_version_compliance():
    """Test that metadata complies with current schema version."""
    
    try:
        with open("ablation_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("⚠️ No metadata file found, skipping schema test")
        return
    
    # Check schema version
    schema_version = metadata.get("schema_version")
    assert schema_version in ["1.0", "1.1"], f"Invalid schema version: {schema_version}"
    
    # Check required fields for 1.1
    if schema_version == "1.1":
        assert "run_name" in metadata, "Schema 1.1 requires run_name field"
        assert isinstance(metadata["run_name"], str), "run_name must be string"
        assert len(metadata["run_name"]) > 0, "run_name cannot be empty"
    
    # Check validation_results
    assert "validation_results" in metadata, "Missing validation_results"
    val_results = metadata["validation_results"]
    assert "total_triples" in val_results, "Missing total_triples"
    assert val_results["total_triples"] >= 1000, "total_triples too small"
    
    print(f"✅ Schema version {schema_version} compliance verified")

def test_production_checksum_validation():
    """Test that production runs have valid checksums."""
    
    try:
        with open("ablation_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("⚠️ No metadata file found, skipping checksum test")
        return
    
    run_name = metadata.get("run_name", "")
    
    if run_name.startswith("production"):
        # Production runs must have schema 1.1
        assert metadata.get("schema_version") == "1.1", f"Production run {run_name} must use schema 1.1"
        
        # Production runs must have checksums for all artifacts
        artifacts = metadata.get("artifacts", {})
        checksums = metadata.get("artifact_checksums", {})
        
        for artifact_type, path in artifacts.items():
            checksum = checksums.get(artifact_type)
            assert checksum is not None, f"Production run {run_name} missing checksum for {artifact_type}"
            assert checksum.startswith("sha256:"), f"Invalid checksum format for {artifact_type}: {checksum}"
            
            # Verify checksum is not placeholder
            assert len(checksum) > 10, f"Checksum too short (likely placeholder) for {artifact_type}"
        
        print(f"✅ Production run {run_name} has valid checksums for all artifacts")
    else:
        print(f"✅ Non-production run {run_name}, checksum validation skipped")

if __name__ == "__main__":
    test_production_run_no_placeholders()
    test_schema_version_compliance()
    test_production_checksum_validation()
