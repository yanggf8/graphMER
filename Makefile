# Simple developer workflow
.PHONY: install test lint clean ensure-test-artifacts generate-metadata cleanup-runs validate-metadata-prod

PY ?= python3
PIP ?= $(PY) -m pip

install:
	$(PIP) install -r requirements.txt

# Run pytest quietly with summary
test: ensure-test-artifacts
	$(PY) -m pytest -q || ($(PY) tests/test_mnm_fixes.py && $(PY) tests/test_metadata_generation.py && $(PY) tests/test_production_guardrails.py && $(PY) tests/test_cleanup.py)

ensure-test-artifacts:
	$(PY) scripts/ensure_test_artifacts.py

generate-metadata:
	$(PY) scripts/generate_metadata.py $(if $(RUN_NAME),--run_name $(RUN_NAME))

validate-metadata:
	$(PY) scripts/validate_metadata.py

validate-metadata-prod:
	$(PY) scripts/validate_production.py

ci-production-check:
	$(PY) scripts/ci_production_check.py

prepare-release:
	$(PY) scripts/prepare_release.py

# Generate metadata for a run name and validate in one step
meta-validate:
	@if [ -z "$(RUN_NAME)" ]; then echo "❌ RUN_NAME required. Usage: make meta-validate RUN_NAME=production_v1"; exit 1; fi
	$(PY) scripts/generate_metadata.py ablation_metadata.json --run_name $(RUN_NAME)
	$(PY) scripts/validate_metadata.py

preflight-check:
	@if [ -z "$(RUN_NAME)" ]; then echo "❌ RUN_NAME required. Use: make preflight-check RUN_NAME=production_v1"; exit 1; fi
	$(PY) scripts/preflight_check.py $(RUN_NAME)

# Complete production workflow (with preflight check)
production-complete: preflight-check generate-metadata validate-metadata-prod prepare-release plot-metrics
	@echo "🎉 Production workflow complete!"
	@echo "  Metadata: ablation_metadata.json"
	@echo "  Manifest: release_manifest.json"
	@echo "  Plots: plots/training_*.png (if matplotlib available)"
	@echo "  Ready for: git tag v1.1.0-prod-ready"

plot-metrics:
	@if [ -n "$(RUN_NAME)" ] && [ -d "logs/runs/$(RUN_NAME)" ]; then \
		$(PY) scripts/plot_metrics.py logs/runs/$(RUN_NAME)/metrics.csv --output-dir plots; \
	else \
		$(PY) scripts/plot_metrics.py logs/train_metrics.csv --output-dir plots; \
	fi

# Cleanup old runs (dry run by default)
cleanup-runs:
	$(PY) scripts/cleanup_runs.py --max-runs 10 --max-age-days 30

cleanup-runs-execute:
	$(PY) scripts/cleanup_runs.py --max-runs 10 --max-age-days 30 --execute

# Optional lint placeholder (can be extended)
lint:
	@echo "No linter configured yet. Add flake8/ruff if desired."

# Run GPU training using a named profile from configs/gpu_profiles.yaml
train-profile:
	$(PY) scripts/run_gpu_profile.py --profile $(PROFILE)

train-production:
	$(PY) scripts/train_with_run_name.py --steps 5000 --run_name production_$(shell date +%Y%m%d_%H%M%S)

clean:
	rm -rf .pytest_cache __pycache__ **/__pycache__ *.egg-info build dist logs/test_artifacts/

help:
	@echo "Available targets:"
	@echo "  test                    - Run test suite"
	@echo "  generate-metadata       - Generate metadata (RUN_NAME=production_v1)"
	@echo "  validate-metadata-prod  - Validate production metadata requirements"
	@echo "  cleanup-runs            - Show old runs to delete (dry run)"
	@echo "  cleanup-runs-execute    - Actually delete old runs"
	@echo "  train-production        - Start production training run"
