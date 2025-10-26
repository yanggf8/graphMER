#!/usr/bin/env python3
"""Cleanup old training runs for storage management."""

import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def get_run_info(run_dir):
    """Get run information for cleanup decisions."""
    try:
        # Try to get creation time from metadata
        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            return {
                'name': run_dir.name,
                'path': run_dir,
                'date': datetime.fromisoformat(metadata.get('validation_date', '1970-01-01')),
                'size_mb': sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file()) / (1024*1024)
            }
        else:
            # Fallback to directory modification time
            return {
                'name': run_dir.name,
                'path': run_dir,
                'date': datetime.fromtimestamp(run_dir.stat().st_mtime),
                'size_mb': sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file()) / (1024*1024)
            }
    except Exception as e:
        print(f"⚠️ Error reading run info for {run_dir}: {e}")
        return None

def cleanup_runs(max_runs=None, max_age_days=None, max_size_gb=None, dry_run=True, protect_production=True):
    """Clean up old training runs based on retention policy."""
    
    runs_dir = Path("logs/runs")
    if not runs_dir.exists():
        print("No runs directory found")
        return
    
    # Get all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print("No runs found")
        return
    
    # Get run info
    runs = []
    for run_dir in run_dirs:
        info = get_run_info(run_dir)
        if info:
            runs.append(info)
    
    # Sort by date (newest first)
    runs.sort(key=lambda x: x['date'], reverse=True)
    
    total_size_gb = sum(run['size_mb'] for run in runs) / 1024
    print(f"Found {len(runs)} runs, total size: {total_size_gb:.1f} GB")
    
    # Determine what to delete
    to_delete = []
    
    # Age-based cleanup
    if max_age_days:
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        for run in runs:
            if run['date'] < cutoff_date:
                if protect_production and run['name'].startswith('production'):
                    print(f"🔒 Protecting production run: {run['name']}")
                    continue
                to_delete.append(run)
    
    # Count-based cleanup (keep newest N)
    if max_runs and len(runs) > max_runs:
        candidates = runs[max_runs:]
        for run in candidates:
            if protect_production and run['name'].startswith('production'):
                print(f"🔒 Protecting production run: {run['name']}")
                continue
            if run not in to_delete:
                to_delete.append(run)
    
    # Size-based cleanup (remove oldest until under limit)
    if max_size_gb:
        current_size = total_size_gb
        for run in reversed(runs):  # Start with oldest
            if current_size <= max_size_gb:
                break
            if protect_production and run['name'].startswith('production'):
                continue
            if run not in to_delete:
                to_delete.append(run)
                current_size -= run['size_mb'] / 1024
    
    # Remove duplicates and sort by date (use path for deduplication)
    seen_paths = set()
    unique_to_delete = []
    for run in to_delete:
        if run['path'] not in seen_paths:
            unique_to_delete.append(run)
            seen_paths.add(run['path'])
    
    to_delete = unique_to_delete
    to_delete.sort(key=lambda x: x['date'])
    
    if not to_delete:
        print("✅ No runs to delete")
        return
    
    # Show what will be deleted
    print(f"\n{'DRY RUN: ' if dry_run else ''}Will delete {len(to_delete)} runs:")
    total_freed_gb = 0
    for run in to_delete:
        total_freed_gb += run['size_mb'] / 1024
        print(f"  - {run['name']} ({run['date'].strftime('%Y-%m-%d')}, {run['size_mb']:.1f} MB)")
    
    print(f"\nTotal space to free: {total_freed_gb:.1f} GB")
    
    # Actually delete if not dry run
    if not dry_run:
        for run in to_delete:
            try:
                shutil.rmtree(run['path'])
                print(f"🗑️ Deleted: {run['name']}")
            except Exception as e:
                print(f"❌ Failed to delete {run['name']}: {e}")
    else:
        print("\n💡 Use --execute to actually delete these runs")

def main():
    parser = argparse.ArgumentParser(description="Clean up old training runs")
    parser.add_argument("--max-runs", type=int, help="Keep only N newest runs")
    parser.add_argument("--max-age-days", type=int, help="Delete runs older than N days")
    parser.add_argument("--max-size-gb", type=float, help="Keep total size under N GB")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry run)")
    parser.add_argument("--no-protect-production", action="store_true", help="Don't protect production runs")
    
    args = parser.parse_args()
    
    if not any([args.max_runs, args.max_age_days, args.max_size_gb]):
        print("❌ Must specify at least one cleanup criterion")
        parser.print_help()
        return
    
    cleanup_runs(
        max_runs=args.max_runs,
        max_age_days=args.max_age_days,
        max_size_gb=args.max_size_gb,
        dry_run=not args.execute,
        protect_production=not args.no_protect_production
    )

if __name__ == "__main__":
    main()
