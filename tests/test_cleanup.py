"""Test cleanup script functionality."""
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_cleanup_dry_run_vs_execute():
    """Test cleanup script dry run vs execute behavior."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock runs directory
        runs_dir = tmpdir / "logs/runs"
        runs_dir.mkdir(parents=True)
        
        # Create test runs with different ages
        old_date = datetime.now() - timedelta(days=40)
        recent_date = datetime.now() - timedelta(days=5)
        
        # Old run (should be deleted)
        old_run = runs_dir / "old_run"
        old_run.mkdir()
        (old_run / "metadata.json").write_text(json.dumps({
            "validation_date": old_date.isoformat(),
            "run_name": "old_run"
        }))
        (old_run / "data.txt").write_text("old data")
        
        # Recent run (should be kept)
        recent_run = runs_dir / "recent_run"
        recent_run.mkdir()
        (recent_run / "metadata.json").write_text(json.dumps({
            "validation_date": recent_date.isoformat(),
            "run_name": "recent_run"
        }))
        (recent_run / "data.txt").write_text("recent data")
        
        # Production run (should be protected)
        prod_run = runs_dir / "production_test"
        prod_run.mkdir()
        (prod_run / "metadata.json").write_text(json.dumps({
            "validation_date": old_date.isoformat(),
            "run_name": "production_test"
        }))
        (prod_run / "data.txt").write_text("production data")
        
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            from scripts.cleanup_runs import cleanup_runs
            
            # Test dry run (should not delete anything)
            cleanup_runs(max_age_days=30, dry_run=True, protect_production=True)
            
            assert old_run.exists(), "Dry run should not delete files"
            assert recent_run.exists(), "Recent run should be kept"
            assert prod_run.exists(), "Production run should be protected"
            
            # Test execute (should delete old non-production run)
            cleanup_runs(max_age_days=30, dry_run=False, protect_production=True)
            
            assert not old_run.exists(), "Old run should be deleted"
            assert recent_run.exists(), "Recent run should be kept"
            assert prod_run.exists(), "Production run should be protected"
            
            print("✅ Cleanup dry run vs execute test passed")
            
        finally:
            os.chdir(original_cwd)

def test_cleanup_protection_patterns():
    """Test that cleanup protects multiple patterns."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        runs_dir = tmpdir / "logs/runs"
        runs_dir.mkdir(parents=True)
        
        old_date = datetime.now() - timedelta(days=40)
        
        # Create runs with different protection patterns
        runs_to_create = [
            ("production_v1", True),    # Should be protected
            ("important_baseline", False),  # Currently not protected (could be enhanced)
            ("test_run_old", False),    # Should be deleted
        ]
        
        for run_name, should_protect in runs_to_create:
            run_dir = runs_dir / run_name
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(json.dumps({
                "validation_date": old_date.isoformat(),
                "run_name": run_name
            }))
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            from scripts.cleanup_runs import cleanup_runs
            
            # Test protection (only production_* currently protected)
            cleanup_runs(max_age_days=30, dry_run=False, protect_production=True)
            
            assert (runs_dir / "production_v1").exists(), "Production run should be protected"
            assert not (runs_dir / "test_run_old").exists(), "Test run should be deleted"
            # Note: important_baseline would be deleted with current logic
            
            print("✅ Cleanup protection patterns test passed")
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    test_cleanup_dry_run_vs_execute()
    test_cleanup_protection_patterns()
