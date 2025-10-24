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
        "--max_samples", str(profile["max_samples"]),
        "--micro_batch_size", str(profile["micro_batch_size"]),
        "--grad_accum_steps", str(profile["grad_accum_steps"])
    ]
    
    if profile.get("amp", False):
        cmd.append("--amp")
    
    if "save_every_steps" in profile:
        # Note: This would need to be added to train_v2.py if not present
        pass
    
    print(f"Executing: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(description="Run GPU training with predefined profiles")
    parser.add_argument("--profile", required=True, help="Profile name (e.g., 406016G)")
    parser.add_argument("--steps", type=int, help="Override steps")
    parser.add_argument("--micro_batch_size", type=int, help="Override micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, help="Override grad accumulation steps")
    parser.add_argument("--save_every_steps", type=int, help="Override save frequency")
    parser.add_argument("--extra", action="store_true", help="Enable extra logging")
    
    args = parser.parse_args()
    
    # Build overrides dict
    overrides = {}
    for key in ["steps", "micro_batch_size", "grad_accum_steps", "save_every_steps"]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    
    return run_profile(args.profile, overrides)

if __name__ == "__main__":
    sys.exit(main())
