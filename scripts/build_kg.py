#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Build a tiny seed KG from local sample files (offline).

def main():
    spec_path = Path("docs/specs/data_spec.yaml")
    spec = yaml.safe_load(spec_path.read_text())
    print("Data sources:", spec.get("sources", {}))
    print("Triples schema:", spec.get("triples", {}).get("schema", {}))

    # Auto-discover Python files with filtering
    from src.kg.builder import build_seed_kg_from_python_files  # type: ignore
    from pathlib import Path as P

    root = P("data/raw/python_samples")
    samples = sorted(root.rglob("*.py"))  # Recursive discovery
    
    # Filter out unwanted files
    samples = [
        p for p in samples 
        if p.is_file() 
        and p.stat().st_size < 50000  # Skip files >50KB
        and not any(skip in str(p) for skip in ['__pycache__', '.git', 'test_'])  # Skip test/cache files
    ]
    
    print(f"Discovered {len(samples)} Python files")
    if not samples:
        print("No Python files found, creating minimal sample")
        sample_file = root / "minimal.py"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text("def hello(): pass\nclass Test: pass")
        samples = [sample_file]
    
    out = P("data/kg/seed_python.jsonl")
    build_seed_kg_from_python_files(samples, out)
    print("Wrote:", out)

    # Validate against ontology
    ent = out.with_suffix(".entities.jsonl")
    from src.ontology.kg_validator import validate_kg
    res = validate_kg(out, ent, Path("docs/specs/ontology_spec.yaml"))
    print("Validation:", res)

if __name__ == "__main__":
    main()
