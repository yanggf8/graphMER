from __future__ import annotations
import yaml
from pathlib import Path

REQUIRED_TOP_LEVEL = {"version", "entities", "relations", "constraints", "triple_schema"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_ontology_spec(path: str | Path) -> tuple[bool, list[str]]:
    path = Path(path)
    spec = load_yaml(path)
    errors: list[str] = []

    missing = REQUIRED_TOP_LEVEL - set(spec.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")

    # Minimal structural checks
    for key in ("entities", "relations"):
        if key in spec and not isinstance(spec[key], dict):
            errors.append(f"{key} must be a mapping")

    # Check a few relations for domain/range presence
    relations = spec.get("relations", {})
    for rname, rdef in relations.items():
        if not isinstance(rdef, dict):
            errors.append(f"Relation {rname} must be a mapping")
            continue
        if "domain" not in rdef or "range" not in rdef:
            errors.append(f"Relation {rname} must have domain and range")

    return (len(errors) == 0, errors)


if __name__ == "__main__":
    ok, errs = validate_ontology_spec("docs/specs/ontology_spec.yaml")
    if not ok:
        print("Ontology spec validation FAILED:")
        for e in errs:
            print(" -", e)
        raise SystemExit(1)
    print("Ontology spec validation OK")
