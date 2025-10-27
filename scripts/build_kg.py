#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Build a seed KG from local sample files (offline).

def main():
    spec_path = Path("docs/specs/data_spec.yaml")
    spec = yaml.safe_load(spec_path.read_text())
    print("Data sources:", spec.get("sources", {}))
    print("Triples schema:", spec.get("triples", {}).get("schema", {}))

    # Auto-discover source files with filtering
    from src.kg.builder import build_seed_kg_from_files  # type: ignore
    from pathlib import Path as P

    py_root = P("data/raw/python_samples")
    java_root = P("data/raw/java_samples")
    js_root = P("data/raw/js_samples")
    
    py_samples = sorted(py_root.rglob("*.py"))
    java_samples = sorted(java_root.rglob("*.java"))
    js_samples = sorted(js_root.rglob("*.js"))
    
    samples = py_samples + java_samples + js_samples
    
    # Filter out unwanted files
    samples = [
        p for p in samples 
        if p.is_file() 
        and p.stat().st_size < 50000  # Skip files >50KB
        and not any(skip in str(p) for skip in ['__pycache__', '.git', 'test_'])  # Skip test/cache files
    ]
    
    print(f"Discovered {len(samples)} source files (Python, Java, and JavaScript)")
    if not samples:
        print("No source files found, exiting.")
        return
    
    out = P("data/kg/seed_multilang.jsonl")
    build_seed_kg_from_files(samples, out)
    print("Wrote:", out)

    # Validate against ontology
    ent = out.with_suffix(".entities.jsonl")
    from src.ontology.kg_validator import validate_kg
    res = validate_kg(out, ent, Path("docs/specs/ontology_spec.yaml"))
    print("Validation:", res)

if __name__ == "__main__":
    main()
