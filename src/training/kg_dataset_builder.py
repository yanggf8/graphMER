from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import json

from .dataset import LeafyChainDataset, simple_tokenize


def tails_from_triples(triples_path: Path, limit: int = 16) -> List[Tuple[str, List[str]]]:
    leaves: List[Tuple[str, List[str]]] = []
    count = 0
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rel = rec.get("relation")
            tail = rec.get("tail")
            if not rel or not tail:
                continue
            # split tail by dot to keep identifiers readable
            toks = []
            for part in str(tail).split("."):
                toks.extend(simple_tokenize(part))
            leaves.append((rel, toks))
            count += 1
            if count >= limit:
                break
    return leaves


def chunk_leaves(leaves: List[Tuple[str, List[str]]], chunk_size: int = 2) -> List[List[Tuple[str, List[str]]]]:
    return [leaves[i:i+chunk_size] for i in range(0, len(leaves), chunk_size)]


def build_dataset_from_kg(triples_path: Path, code_path: Path, max_seq_len: int = 128, limit: int = 16, chunk_size: int = 2) -> LeafyChainDataset:
    text = code_path.read_text(encoding="utf-8")
    leaves_all = tails_from_triples(triples_path, limit=limit)
    groups = chunk_leaves(leaves_all, chunk_size=chunk_size)
    if not groups:
        groups = [[]]
    texts = [text for _ in groups]
    return LeafyChainDataset(texts, groups, max_seq_len=max_seq_len)
