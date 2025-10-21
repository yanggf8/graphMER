#!/usr/bin/env python3
"""
Script to update ablation_metadata.json with current run parameters
"""
import json
from pathlib import Path
import sys
import argparse
import platform
import subprocess
from datetime import datetime
import torch

def update_ablation_metadata(config_path, seed, steps, device_type, run_type="tpu"):
    """
    Update ablation_metadata.json with current run parameters
    """
    # Load existing metadata
    metadata_path = Path("ablation_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "schema_version": "1.0",
            "validation_date": datetime.now().isoformat(),
            "commit_hash": "no-git",
            "dataset_manifest": "data/kg/manifest.json",
            "dataset_hash": "sha256:97dec5a7845ec2dc7618c6ee8408e96c370368701e2c9c51c7e074032aa293ee",
            "ablation_config": "configs/train_cpu.yaml",
            "random_seed": 42,
            "environment": {},
            "artifacts": {},
            "validation_results": {}
        }
    
    # Update with current run parameters
    metadata["validation_date"] = datetime.now().isoformat()
    metadata["ablation_config"] = str(config_path)
    metadata["random_seed"] = seed
    metadata["run_parameters"] = {
        "steps": steps,
        "device_type": device_type,
        "run_type": run_type
    }
    
    # Update environment section
    if "environment" not in metadata:
        metadata["environment"] = {}
    metadata["environment"].update({
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    })
    
    # Try to get git commit hash
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        metadata["commit_hash"] = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        metadata["commit_hash"] = "no-git"
    
    # Attempt to get XLA info if on TPU
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        metadata["environment"]["xla_available"] = True
        metadata["environment"]["xla_version"] = xm.get_xla_version() if hasattr(xm, 'get_xla_version') else "unknown"
        metadata["environment"]["torch_xla_version"] = xm.get_torch_xla_version() if hasattr(xm, 'get_torch_xla_version') else "unknown"
        
        # Add XLA-specific information
        metadata["xla_info"] = {}
        try:
            # Get XLA device information
            if hasattr(xm, 'get_xla_supported_devices'):
                metadata["xla_info"]["supported_devices"] = xm.get_xla_supported_devices()
            # Get world size if available
            if hasattr(xm, 'xrt_world_size'):
                metadata["xla_info"]["world_size"] = xm.xrt_world_size()
            else:
                metadata["xla_info"]["world_size"] = 1  # Default to 1 if not distributed
            # Get ordinal (device ID) if available
            if hasattr(xm, 'get_ordinal'):
                metadata["xla_info"]["ordinal"] = xm.get_ordinal()
        except Exception as e:
            print(f"Warning: Could not get detailed XLA info: {e}")
            metadata["xla_info"]["world_size"] = 1
            
        # Try to get TPU topology info
        try:
            import os
            tpu_topology = os.environ.get('TPU_TOPOLOGY', 'unknown')
            metadata["xla_info"]["tpu_topology"] = tpu_topology
            
            # Get TPU version from XLA
            if hasattr(xr, 'global_runtime_device_count'):
                metadata["xla_info"]["global_device_count"] = xr.global_runtime_device_count()
        except Exception as e:
            print(f"Warning: Could not get TPU topology info: {e}")
            
    except ImportError:
        metadata["environment"]["xla_available"] = False
        metadata["xla_info"] = {
            "world_size": 1,
            "tpu_topology": "none",
            "global_device_count": 0
        }
    
    # Save updated metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated ablation metadata at {metadata_path}")
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Update ablation metadata for current run")
    parser.add_argument("--config", type=str, required=True, help="Config file used for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device type (cpu or tpu)")
    parser.add_argument("--run-type", type=str, default="training", help="Type of run")
    
    args = parser.parse_args()
    
    metadata_path = update_ablation_metadata(
        config_path=args.config,
        seed=args.seed,
        steps=args.steps,
        device_type=args.device,
        run_type=args.run_type
    )
    
    print(f"Metadata updated successfully at {metadata_path}")


if __name__ == "__main__":
    main()