#!/usr/bin/env python3
"""GPU profile runner - executes predefined training configurations."""
import argparse
import yaml
import subprocess
import sys
from pathlib import Path

def load_profiles():
    """Load GPU profiles from config file."""
    profile_path = Path(__file__).parent.parent / "configs" / "gpu_profiles.yaml"
    with open(profile_path) as f:
        return yaml.safe_load(f)

def run_profile(profile_name, overrides=None):
    """Run training with specified profile and optional overrides."""
    profiles = load_profiles()
    
    if profile_name not in profiles:
        print(f"Error: Profile '{profile_name}' not found.")
        print(f"Available profiles: {list(profiles.keys())}")
        return 1
    
    profile = profiles[profile_name].copy()
    print(f"Running profile: {profile_name}")
    print(f"Description: {profile.get('description', 'No description')}")
    
    # Apply overrides
    if overrides:
        profile.update(overrides)
    
    # Build command
    cmd = [
        "python3", "scripts/train_v2.py",
        "--config", profile["config"],
        "--steps", str(profile["steps"]),
        "--max_samples", str(profile.get("max_samples", 0) if profile.get("max_samples", None) is not None else 0),
        "--micro_batch_size", str(profile["micro_batch_size"]),
        "--grad_accum_steps", str(profile["grad_accum_steps"])
    ]
    
    if profile.get("amp", False):
        cmd.append("--amp")
    
    if "save_every_steps" in profile and profile["save_every_steps"]:
        cmd.extend(["--save_every_steps", str(profile["save_every_steps"])])

    # Optional tuning flags
    flag_mapping = {
        "warmup_steps": "--warmup_steps",
        "clip_grad": "--clip_grad",
        "mnm_weight_ramp": "--mnm_weight_ramp",
        "mlm_weight": "--mlm_weight",
        "mnm_weight": "--mnm_weight",
        "log_mnm_debug": "--log_mnm_debug",
        "max_code_files": "--max_code_files",
    }
    for key, flag in flag_mapping.items():
        if key in profile and profile[key] is not None:
            cmd.extend([flag, str(profile[key])])

    if profile.get("use_full_kg"):
        cmd.append("--use_full_kg")

    extra_args = profile.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list):
            raise TypeError(f"extra_args for profile '{profile_name}' must be a list")
        cmd.extend(str(arg) for arg in extra_args)
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(description="Run GPU training with predefined profiles")
    parser.add_argument("--profile", required=True, help="Profile name (e.g., 406016G)")
    parser.add_argument("--steps", type=int, help="Override steps")
    parser.add_argument("--max_samples", type=int, help="Override max samples")
    parser.add_argument("--micro_batch_size", type=int, help="Override micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, help="Override grad accumulation steps")
    parser.add_argument("--save_every_steps", type=int, help="Override save frequency")
    parser.add_argument("--extra", action="store_true", help="Enable extra logging")
    
    args = parser.parse_args()
    
    # Build overrides dict
    overrides = {}
    for key in ["steps", "max_samples", "micro_batch_size", "grad_accum_steps", "save_every_steps"]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    
    return run_profile(args.profile, overrides)

if __name__ == "__main__":
    sys.exit(main())
