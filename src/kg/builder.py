from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List
from pathlib import Path
import json

from src.parsing.python_parser import PythonCodeParser
from src.parsing.java_parser import JavaCodeParser
from src.parsing.js_parser import JavaScriptCodeParser

def build_seed_kg_from_files(files: List[Path], out_path: Path) -> None:
    triples_out = []
    entities_out = {}
    for f in files:
        code = f.read_text(encoding="utf-8")
        if f.suffix == '.py':
            module = f.stem  # simple module name
            parser = PythonCodeParser(str(f), module)
            entities, triples = parser.parse(code)
        elif f.suffix == '.java':
            parser = JavaCodeParser(str(f))
            entities, triples = parser.parse(code)
        elif f.suffix == '.js':
            parser = JavaScriptCodeParser(str(f))
            entities, triples = parser.parse(code)
        else:
            continue # Skip unsupported file types

        for e in entities.values():
            entities_out[e.id] = {"id": e.id, "type": e.type, "file": e.file}
        for t in triples:
            triples_out.append({
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "qualifiers": t.qualifiers,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for trip in triples_out:
            f.write(json.dumps(trip, ensure_ascii=False) + "\n")

    # Also write entities
    ent_path = out_path.with_suffix(".entities.jsonl")
    with ent_path.open("w", encoding="utf-8") as f:
        for e in entities_out.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
