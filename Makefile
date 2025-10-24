# Simple developer workflow
.PHONY: install test lint clean

PY ?= python3
PIP ?= $(PY) -m pip

install:
	$(PIP) install -r requirements.txt

# Run pytest quietly with summary
test:
	$(PY) -m pytest -q

# Optional lint placeholder (can be extended)
lint:
	@echo "No linter configured yet. Add flake8/ruff if desired."

# Run GPU training using a named profile from configs/gpu_profiles.yaml
train-profile:
	$(PY) scripts/run_gpu_profile.py --profile $(PROFILE)

clean:
	rm -rf .pytest_cache __pycache__ **/__pycache__ *.egg-info build dist
