#!/usr/bin/env python3
"""Enhanced KG builder with recursive discovery and manifests (multi-language)."""
import json
import hashlib
from pathlib import Path
import argparse
import time
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def discover_files(root_dir: Path, max_files: int = 500) -> tuple[list[Path], list[Path]]:
    """Discover Python and Java files with filtering"""
    py_files = []
    java_files = []
    skip_patterns = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', '.venv'}

    for py_file in root_dir.rglob("*.py"):
        if any(part in skip_patterns for part in py_file.parts):
            continue
        if py_file.stat().st_size > 100_000:
            continue
        py_files.append(py_file)
        if len(py_files) >= max_files // 2:
            break

    for java_file in root_dir.rglob("*.java"):
        if any(part in skip_patterns for part in java_file.parts):
            continue
        if java_file.stat().st_size > 100_000:
            continue
        java_files.append(java_file)
        if len(java_files) >= max_files // 2:
            break

    return py_files, java_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="data/raw")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/kg")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Discover files
    py_files, java_files = discover_files(source_dir, args.max_files)
    print(f"Discovered {len(py_files)} Python files, {len(java_files)} Java files")

    # Import after discovery to avoid overhead
    from src.kg.builder import build_seed_kg_from_files
    from src.ontology.kg_validator import validate_kg
    # Java parsing is optional; skip if dependency is unavailable
    java_available = True
    JavaParser = None
    try:
        from src.parsing.java_parser import JavaParser  # requires javalang
    except Exception as e:
        print(f"Java parsing disabled: {e}")
        java_available = False

    file_hashes: dict[str, str] = {}
    all_triples_flat: list[dict] = []  # flat schema: head(str), relation(str), tail(str), qualifiers(dict)
    entities: dict[str, dict] = {}

    # Process Python files
    if py_files:
        py_triples_file = output_dir / "temp_python.jsonl"
        build_seed_kg_from_files(py_files, py_triples_file)

        # Read Python triples (already flat)
        if py_triples_file.exists():
            with open(py_triples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_triples_flat.append(json.loads(line))
        # Read Python entities and merge types
        py_entities_file = py_triples_file.with_suffix('.entities.jsonl')
        if py_entities_file.exists():
            with open(py_entities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    eid = rec.get('id')
                    if eid:
                        entities[eid] = {k: rec[k] for k in ('id', 'type') if k in rec}
        # Clean up temp files
        if py_triples_file.exists():
            py_triples_file.unlink()
        if py_entities_file.exists():
            py_entities_file.unlink()

        for f in py_files:
            file_hashes[str(f)] = compute_file_hash(f)

    # Process Java files
    if java_files and java_available and JavaParser is not None:
        java_parser = JavaParser()
        for java_file in java_files:
            try:
                code = java_file.read_text(encoding='utf-8')
                java_triples = java_parser.parse(code, str(java_file))
                # Normalize Java triples to flat schema and accumulate entities
                for t in java_triples:
                    try:
                        h = t['head']['id']
                        ht = t['head'].get('type')
                        r = t['relation']['type'] if isinstance(t.get('relation'), dict) else t.get('relation')
                        ta = t['tail']['id']
                        tt = t['tail'].get('type')
                        q = t.get('qualifiers', {})
                        all_triples_flat.append({
                            'head': h,
                            'relation': r,
                            'tail': ta,
                            'qualifiers': q
                        })
                        if h and ht:
                            entities[h] = {'id': h, 'type': ht}
                        if ta and tt:
                            entities[ta] = {'id': ta, 'type': tt}
                    except Exception as e:
                        print(f"Skipping malformed Java triple from {java_file}: {t} ({e})")
                file_hashes[str(java_file)] = compute_file_hash(java_file)
            except Exception as e:
                print(f"Error parsing {java_file}: {e}")

    # Write combined triples (flat schema)
    triples_file = output_dir / "enhanced_multilang.jsonl"
    with open(triples_file, 'w', encoding='utf-8') as f:
        for triple in all_triples_flat:
            f.write(json.dumps(triple, ensure_ascii=False) + '\n')

    # Write combined entities (types for both languages)
    entities_file = output_dir / "enhanced_multilang.entities.jsonl"
    with open(entities_file, 'w', encoding='utf-8') as f:
        for ent in entities.values():
            f.write(json.dumps(ent, ensure_ascii=False) + '\n')

    triple_count = len(all_triples_flat)

    # Validate
    ontology_file = Path("docs/specs/ontology_spec.yaml")
    validation_result = validate_kg(triples_file, entities_file, ontology_file)

    # Create manifest
    manifest = {
        "version": "1.0",
        "timestamp": time.time(),
        "build_duration_sec": time.time() - start_time,
        "source_dir": str(source_dir),
        "files_processed": {"python": len(py_files), "java": len(java_files)},
        "total_triples": triple_count,
        "validation": validation_result,
        "file_hashes": file_hashes,
        "ontology_version": "0.1.0",
        "outputs": {
            "triples": str(triples_file),
            "entities": str(entities_file)
        }
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"KG build complete: {triple_count} triples in {time.time() - start_time:.1f}s")
    print(f"Validation: {validation_result}")
    print(f"Manifest: {manifest_file}")

if __name__ == "__main__":
    main()
