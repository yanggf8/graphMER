from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Set
import yaml
import json


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_entities(path: Path) -> Dict[str, Dict[str, str]]:
    entities: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            entities[rec["id"]] = rec
    return entities


def load_triples(path: Path) -> List[Dict]:
    triples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            triples.append(json.loads(line))
    return triples


def build_relation_schema(ontology: dict) -> Dict[str, Tuple[Set[str], Set[str]]]:
    schema: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for rname, rdef in ontology.get("relations", {}).items():
        dom = set(rdef.get("domain", []) or [])
        rng = set(rdef.get("range", []) or [])
        schema[rname] = (dom, rng)
    return schema


def validate_domain_range(triples: List[Dict], entities: Dict[str, Dict[str, str]], schema: Dict[str, Tuple[Set[str], Set[str]]]) -> Tuple[int, int, List[Dict]]:
    total = 0
    ok = 0
    failures: List[Dict] = []
    for t in triples:
        total += 1
        r = t.get("relation")
        head_id = t.get("head")
        tail_id = t.get("tail")
        head = entities.get(head_id, {})
        tail = entities.get(tail_id, {})
        head_type = head.get("type")
        tail_type = tail.get("type")
        dom_rng = schema.get(r)
        if dom_rng is None:
            failures.append({"triple": t, "reason": "unknown_relation", "head_type": head_type, "tail_type": tail_type})
            continue
        dom, rng = dom_rng
        if head_type in dom and tail_type in rng:
            ok += 1
        else:
            failures.append({"triple": t, "reason": "domain_range_mismatch", "expected_domain": list(dom), "expected_range": list(rng), "head_type": head_type, "tail_type": tail_type})
    return ok, total, failures


def validate_acyclicity_inherits(triples: List[Dict]) -> bool:
    # Build graph for inherits_from
    graph: Dict[str, List[str]] = {}
    for t in triples:
        if t.get("relation") == "inherits_from":
            h = t.get("head")
            ta = t.get("tail")
            graph.setdefault(h, []).append(ta)
    # DFS cycle detection
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return False  # cycle
        if node in visited:
            return True
        visiting.add(node)
        for nei in graph.get(node, []):
            if not dfs(nei):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(dfs(n) for n in list(graph.keys()))


def validate_kg(triples_path: Path, entities_path: Path, ontology_path: Path) -> Dict[str, object]:
    ontology = load_yaml(ontology_path)
    rel_schema = build_relation_schema(ontology)
    entities = load_entities(entities_path)
    triples = load_triples(triples_path)

    ok_dom, total, failures = validate_domain_range(triples, entities, rel_schema)
    acyclic = validate_acyclicity_inherits(triples)

    result = {
        "domain_range_ok": ok_dom,
        "total_triples": total,
        "domain_range_ratio": (ok_dom / total) if total else 1.0,
        "inherits_acyclic": acyclic,
        "failures_preview": failures[:10],
    }
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("triples", type=str)
    p.add_argument("entities", type=str)
    p.add_argument("ontology", type=str, default="docs/specs/ontology_spec.yaml")
    args = p.parse_args()

    res = validate_kg(Path(args.triples), Path(args.entities), Path(args.ontology))
    print(res)
    # simple gate
    if not res["inherits_acyclic"] or res["domain_range_ratio"] < 0.95:
        raise SystemExit(1)
