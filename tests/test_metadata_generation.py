"""Test metadata generation and artifact management."""
import sys
import json
import tempfile
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_generate_metadata_with_run_name():
    """Test metadata generation with specific run name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock run directory
        run_dir = tmpdir / "logs/runs/test_run"
        run_dir.mkdir(parents=True)
        
        # Create mock artifacts
        metrics_file = run_dir / "metrics.csv"
        metrics_file.write_text(
            "step,total_loss,mlm_loss,mnm_loss,mlm_validation_accuracy,mnm_validation_accuracy\n"
            "100,0.1,0.05,0.05,0.8,0.9\n"
        )
        
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "model_final.pt"
        checkpoint_file.write_bytes(b"mock_checkpoint")
        
        # Create mock KG manifest
        kg_dir = tmpdir / "data/kg"
        kg_dir.mkdir(parents=True)
        manifest_file = kg_dir / "manifest.json"
        manifest_file.write_text(json.dumps({
            "total_triples": 1000,
            "validation": {"domain_range_ratio": 0.95}
        }))
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            from scripts.generate_metadata import generate_metadata
            
            output_file = tmpdir / "test_metadata.json"
            generate_metadata(str(output_file), "test_run")
            
            # Verify metadata
            with open(output_file) as f:
                metadata = json.load(f)
            
            assert metadata["schema_version"] == "1.1"
            assert metadata["run_name"] == "test_run"
            assert "logs/runs/test_run/metrics.csv" in metadata["artifacts"]["metrics_csv"]
            assert metadata["validation_results"]["total_triples"] == 1000
            
            print("✅ Generate metadata with run_name test passed")
            
        finally:
            os.chdir(original_cwd)

def test_ensure_test_artifacts_ci_mode():
    """Test test artifact creation in CI mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        # Set CI environment
        os.environ["CI"] = "true"
        
        try:
            from scripts.ensure_test_artifacts import ensure_test_artifacts
            
            ensure_test_artifacts()
            
            # Verify test artifacts created
            assert Path("logs/test_artifacts/test_metrics.csv").exists()
            assert Path("logs/test_artifacts/test_checkpoint.pt").exists()
            assert Path("ablation_metadata.json").exists()
            
            # Verify metadata points to test artifacts
            with open("ablation_metadata.json") as f:
                metadata = json.load(f)
            
            assert metadata["run_name"] == "test_run"
            assert "test_artifacts" in metadata["artifacts"]["metrics_csv"]
            
            print("✅ Test artifacts CI mode test passed")
            
        finally:
            os.chdir(original_cwd)
            if "CI" in os.environ:
                del os.environ["CI"]

def test_production_run_validation():
    """Test that production runs don't use placeholders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create production run with small files (placeholders)
        run_dir = tmpdir / "logs/runs/production_test"
        run_dir.mkdir(parents=True)
        
        # Small files that should trigger warnings
        (run_dir / "metrics.csv").write_text("small")
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model_final.pt").write_bytes(b"small")
        
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            from scripts.generate_metadata import generate_metadata
            import io
            import contextlib
            
            # Capture output to check for warnings
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                generate_metadata("test_metadata.json", "production_test")
            
            output = f.getvalue()
            assert "WARNING" in output
            assert "Production run using small" in output
            
            print("✅ Production run validation test passed")
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    test_generate_metadata_with_run_name()
    test_ensure_test_artifacts_ci_mode()
    test_production_run_validation()
