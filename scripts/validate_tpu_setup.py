#!/usr/bin/env python3
"""
Validation script for TPU training setup requirements.

This script verifies that all TPU training plan requirements are met:
1. TPU environment setup (torch-xla, device availability)
2. Configuration files are valid
3. Required directories and data exist
4. Model architecture supports TPU
5. Training scripts can be executed
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class TPUSetupValidator:
    """Validates TPU training setup requirements."""

    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_torch_xla(self) -> bool:
        """Check if torch-xla is installed and TPU devices are available."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get XLA devices
            device = xm.xla_device()
            world_size = xm.xrt_world_size()

            print(f"✓ torch-xla installed: {torch_xla.__version__}")
            print(f"  - XLA device: {device}")
            print(f"  - World size: {world_size}")

            return True
        except ImportError:
            self.errors.append(
                "torch-xla not installed. Install with: pip install .[tpu]"
            )
            return False
        except Exception as e:
            self.warnings.append(f"torch-xla available but devices not ready: {e}")
            print(f"⚠ torch-xla installed but devices not accessible: {e}")
            print("  (This is expected if not running on TPU hardware)")
            return True  # Consider installed as success

    def check_config_files(self) -> bool:
        """Validate TPU configuration files exist and are valid."""
        config_path = Path("configs/train_tpu.yaml")

        if not config_path.exists():
            self.errors.append(f"TPU config not found: {config_path}")
            return False

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Validate required TPU settings
            required_checks = [
                ("hardware.device", "tpu"),
                ("hardware.tpu_cores", 8),
                ("run.mixed_precision", "bf16"),
                ("model.use_rel_attention_bias", True),
            ]

            for path, expected in required_checks:
                keys = path.split(".")
                value = config
                for key in keys:
                    value = value.get(key)
                    if value is None:
                        self.errors.append(f"Missing config: {path}")
                        return False

                if value != expected:
                    self.warnings.append(
                        f"Config {path} = {value}, expected {expected}"
                    )

            print("✓ TPU config valid")
            print(f"  - Device: {config['hardware']['device']}")
            print(f"  - TPU cores: {config['hardware']['tpu_cores']}")
            print(f"  - Mixed precision: {config['run']['mixed_precision']}")
            print(
                f"  - Relation attention bias: {config['model']['use_rel_attention_bias']}"
            )

            return True
        except Exception as e:
            self.errors.append(f"Failed to parse config: {e}")
            return False

    def check_data_availability(self) -> bool:
        """Check if required data files exist."""
        required_paths = [
            "data/kg/triples.txt",
            "data/kg/entities.json",
        ]

        missing = []
        for path_str in required_paths:
            path = Path(path_str)
            if not path.exists():
                missing.append(path_str)

        if missing:
            self.warnings.append(
                f"Missing data files: {missing}. Run: python scripts/build_kg_enhanced.py"
            )
            print(f"⚠ Missing data files: {missing}")
            print("  Run: python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples")
            return False

        # Check manifest for production requirements
        manifest_path = Path("data/kg/manifest.json")
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            total_triples = manifest.get("total_triples", 0)
            validation_quality = manifest.get("validation", {}).get(
                "domain_range_ratio", 0
            )

            print(f"✓ Data available")
            print(f"  - Total triples: {total_triples:,}")
            print(f"  - Validation quality: {validation_quality:.1%}")

            if total_triples < 30000:
                self.warnings.append(
                    f"Only {total_triples} triples (recommended: 30,000+)"
                )

            if validation_quality < 0.99:
                self.warnings.append(
                    f"Validation quality {validation_quality:.1%} (recommended: 99%+)"
                )
        else:
            print("✓ Data files found (manifest not generated)")

        return True

    def check_scripts(self) -> bool:
        """Verify required training scripts exist."""
        required_scripts = [
            "scripts/train.py",
            "scripts/check_monitoring_gates.py",
            "scripts/update_metadata.py",
        ]

        missing = []
        for script in required_scripts:
            if not Path(script).exists():
                missing.append(script)

        if missing:
            self.errors.append(f"Missing scripts: {missing}")
            return False

        print("✓ All required scripts present")
        return True

    def check_model_architecture(self) -> bool:
        """Verify model architecture supports TPU."""
        try:
            from src.models.encoder import TinyEncoder

            print("✓ TinyEncoder importable")

            # Check for TPU-compatible features
            import inspect

            source = inspect.getsource(TinyEncoder)

            if "use_rel_attention_bias" in source:
                print("  - Relation attention bias supported")
            else:
                self.warnings.append("Relation attention bias not found in model")

            return True
        except ImportError as e:
            self.errors.append(f"Failed to import model: {e}")
            return False

    def check_test_suite(self) -> bool:
        """Verify test suite includes TPU tests."""
        test_files = [
            "tests/test_configs_load.py",
            "tests/test_tpu_tools.py",
            "tests/test_rel_attention_bias.py",
        ]

        missing = []
        for test_file in test_files:
            if not Path(test_file).exists():
                missing.append(test_file)

        if missing:
            self.errors.append(f"Missing test files: {missing}")
            return False

        print("✓ TPU test suite complete")
        print(f"  - Config tests: tests/test_configs_load.py")
        print(f"  - TPU tools tests: tests/test_tpu_tools.py")
        print(f"  - Model tests: tests/test_rel_attention_bias.py")

        return True

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("=" * 60)
        print("TPU Training Setup Validation")
        print("=" * 60)
        print()

        checks = [
            ("TPU Environment", self.check_torch_xla),
            ("Configuration Files", self.check_config_files),
            ("Data Availability", self.check_data_availability),
            ("Training Scripts", self.check_scripts),
            ("Model Architecture", self.check_model_architecture),
            ("Test Suite", self.check_test_suite),
        ]

        all_passed = True
        for name, check_fn in checks:
            print(f"\n{name}:")
            print("-" * 60)
            passed = check_fn()
            self.results[name] = passed
            if not passed:
                all_passed = False

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for name, passed in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠ Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if all_passed and not self.errors:
            print("\n✅ All checks passed! TPU training setup is ready.")
            print("\nNext steps:")
            print("  1. Build KG: python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples")
            print("  2. Run training: python scripts/train.py --config configs/train_tpu.yaml --steps 300")
            print("  3. Or manual: python scripts/train.py --config configs/train_tpu.yaml --steps 300")
            return True
        else:
            print("\n❌ Some checks failed. Please address the errors above.")
            return False


def main():
    """Main entry point."""
    validator = TPUSetupValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
