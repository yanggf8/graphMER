from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List
from pathlib import Path
import json

from src.parsing.python_parser import PythonCodeParser
# Optional language parsers; allow environments without extra deps
from src.parsing.python_parser import PythonCodeParser
try:
    from src.parsing.js_parser import JavaScriptCodeParser  # type: ignore
    _JS_AVAILABLE = True
except Exception:
    JavaScriptCodeParser = None  # type: ignore
    _JS_AVAILABLE = False

# Java is optional; allow environments without javalang
try:
    from src.parsing.java_parser import JavaCodeParser  # type: ignore
    _JAVA_AVAILABLE = True
except Exception:
    JavaCodeParser = None  # type: ignore
    _JAVA_AVAILABLE = False

def build_seed_kg_from_files(files: List[Path], out_path: Path) -> None:
    triples_out = []
    entities_out = {}
    for f in files:
        code = f.read_text(encoding="utf-8")
        if f.suffix == '.py':
            module = f.stem  # simple module name
            parser = PythonCodeParser(str(f), module)
            entities, triples = parser.parse(code)
        elif f.suffix == '.java' and _JAVA_AVAILABLE and JavaCodeParser is not None:
            parser = JavaCodeParser(str(f))
            entities, triples = parser.parse(code)
        elif f.suffix == '.js' and _JS_AVAILABLE and JavaScriptCodeParser is not None:
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
