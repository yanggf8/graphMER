# Ensure project root is on sys.path so tests can import 'src' and 'scripts'
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure required test artifacts/metadata exist before tests run
@pytest.fixture(scope="session", autouse=True)
def _rovodev_ensure_test_artifacts():
    try:
        from scripts.ensure_test_artifacts import ensure_test_artifacts
        ensure_test_artifacts()
    except Exception as e:
        # Non-fatal: allow tests to proceed; individual tests may skip
        print(f"[conftest] Warning: ensure_test_artifacts failed: {e}")
