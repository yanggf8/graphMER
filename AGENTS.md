# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core Python packages: `models/` for the transformer, `training/` for dataset loaders and losses, `ontology/` for graph validation, and `utils/` for shared helpers.
- `scripts/` bundles automation entry points (`train_v2.py`, `run_gpu_profile.py`, `generate_metadata.py`, etc.); treat these as the public CLI surface.
- `configs/` stores YAML profiles used by `scripts/run_gpu_profile.py` and production jobs; keep run-specific tweaks out of `src/`.
- `tests/` mirrors the runtime layout with focused suites (`test_model.py`, `test_dataset.py`, `test_metadata.py`). Docs, checkpoints, and example assets live in `docs/`, `logs/`, and `data/` respectively.

## Build, Test, and Development Commands
- `make install` installs Python 3.10+ dependencies from `requirements.txt`.
- `make test` runs `pytest -q` after seeding placeholder artifacts via `scripts/ensure_test_artifacts.py`; rely on this target instead of calling pytest directly.
- `python scripts/run_gpu_profile.py --profile 408032G --steps 5000` executes the validated GPU training loop described in `configs/gpu_profiles.yaml`.
- `python scripts/run_gpu_profile.py --profile M2_8C_16G` drives the optimized Apple M2/MPS curriculum using `configs/train_mps.yaml`, full KG sampling, warmup, and gradient clipping baked in.
- `make generate-metadata RUN_NAME=production_YYYYMMDD_HHMMSS` regenerates `ablation_metadata.json`; follow up with `make validate-metadata-prod` before release tags.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints throughout, and module-level docstrings for exported classes/functions. Match existing imports ordering and prefer explicit relative imports inside packages.
- Keep minibatches and tensors named descriptively (e.g., `attn_logits`, `rel_ids`) as in `src/models/encoder.py`; avoid single-letter variables outside tight loops.
- Scripts accept long-form flags; use `--profile`, `--run_name`, etc., and document new options in the script docstring and README sections touching automation.

## Testing Guidelines
- Test with `pytest` (configured via `pytest.ini`); name files and functions `test_*` to ensure discovery.
- Tests rely on generated artifacts in `logs/test_artifacts/`; never commit production checkpoints into tests, and refresh fixtures with `make test` after modifying schemas.
- Add property or regression tests alongside new modules (e.g., extend `tests/test_dataset.py` when altering `dataset_v2` logic) and keep runtime shorter than 2 minutes.

## Commit & Pull Request Guidelines
- Follow the repository’s conventional style: lowercase type prefixes such as `feat:`, `docs:`, `fix:`, or `chore:` summarizing the primary change; group related updates in a single message rather than stacking commits.
- PRs should include: a short changelog-style summary, links to tracked issues or paper requirements, confirmation that `make test` and metadata validation were run, and screenshots or metric snippets when touching training/eval flows.
- Avoid force-pushing over reviewed commits; instead, address review feedback with follow-up commits maintaining the same prefix.

## Metadata & Artifact Management
- Name training runs `production_YYYYMMDD_HHMMSS` for production or `experiment_*` for exploratory work so that automation in `scripts/preflight_check.py` and cleanup guards treat them correctly.
- Keep production artifacts in `logs/checkpoints/` and record SHA256 entries in `ablation_metadata.json`; placeholder artifacts belong only under `logs/test_artifacts/`.
- Before release, execute `make production-complete` to chain preflight checks, metadata generation, validation, and plotting in the expected order.
