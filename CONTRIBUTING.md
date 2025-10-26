# Contributing to GraphMER-SE

## Development Workflow

### Running Tests

```bash
# Run complete test suite (includes artifact setup)
make test

# Tests automatically create placeholder artifacts in logs/test_artifacts/
# This prevents CI failures when training hasn't been run locally
```

**Note**: The test suite uses placeholder artifacts to ensure schema compliance without requiring actual training runs. These are automatically created in `logs/test_artifacts/` and should not be confused with production artifacts.

### Training and Metadata

#### Run Naming Convention

- **Production runs**: Use `production_YYYYMMDD_HHMMSS` format
- **Development runs**: Use descriptive names like `experiment_feature_x`
- **Test runs**: Use `test_` prefix for temporary experiments

#### Generating Metadata

```bash
# For production runs (strict validation)
make generate-metadata RUN_NAME=production_20241025_120000

# For development runs
make generate-metadata RUN_NAME=my_experiment

# Current training (uses logs/train_metrics.csv)
make generate-metadata
```

#### Production Run Requirements

Production runs (`production_*`) have enhanced validation:
- Must use schema version 1.1
- All artifacts must have valid SHA256 checksums
- Artifact files must meet minimum size thresholds
- Cannot reference test placeholder paths

### Artifact Management

#### Cleanup Old Runs

```bash
# Preview what will be deleted (dry run)
make cleanup-runs

# Actually delete old runs (protects production_* runs)
make cleanup-runs-execute
```

#### Verifying Artifacts

```bash
# Check artifact integrity
sha256sum logs/runs/production_v1/metrics.csv | grep $(jq -r '.artifact_checksums.metrics_csv' ablation_metadata.json | cut -d: -f2)
```

## Code Quality

### Before Submitting PRs

1. Run the test suite: `make test`
2. Validate metadata: `make validate-metadata`
3. Ensure no production artifacts in test paths
4. Update documentation if adding new features

### Schema Changes

If modifying `ablation_metadata.json` structure:
1. Update `docs/specs/metadata_schema.json`
2. Bump schema version if breaking changes
3. Add backward compatibility handling
4. Update tests to cover new fields

## CI/CD Notes

- Tests run with placeholder artifacts to avoid training dependencies
- Production runs are protected from automated cleanup
- Metadata validation ensures schema compliance
- All artifacts include integrity checksums
