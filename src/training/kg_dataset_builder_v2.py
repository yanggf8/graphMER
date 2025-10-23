"""Enhanced KG dataset builder that uses ALL available triples."""
from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import json
import random

from .dataset_v2 import LeafyChainDatasetV2


def load_all_triples(triples_path: Path) -> List[Dict]:
    """Load all triples from KG file."""
    triples = []
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                triples.append(rec)
            except json.JSONDecodeError:
                continue
    return triples


def group_triples_by_head(triples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group triples by head entity to create coherent training samples."""
    grouped = {}
    for triple in triples:
        head = triple.get("head", "")
        if not head:
            continue
        if head not in grouped:
            grouped[head] = []
        grouped[head].append(triple)
    return grouped


def create_training_samples(
    grouped_triples: Dict[str, List[Dict]], 
    code_snippets: Dict[str, str],
    max_leaves_per_sample: int = 5
) -> Tuple[List[str], List[List[Tuple[str, List[str]]]]]:
    """Create training samples from grouped triples.
    
    Args:
        grouped_triples: Triples grouped by head entity
        code_snippets: Dict mapping entity names to code text
        max_leaves_per_sample: Max number of relation chains per sample
    
    Returns:
        (code_texts, leaves_per_sample)
    """
    texts = []
    leaves_list = []
    
    for head, triples in grouped_triples.items():
        # Get code for this head entity
        code = code_snippets.get(head, head)
        
        # Convert triples to leaves format
        leaves = []
        for triple in triples[:max_leaves_per_sample]:
            relation = triple.get("relation", "")
            tail = triple.get("tail", "")
            if not relation or not tail:
                continue
            
            # Tokenize tail into parts
            tail_tokens = tail.split(".")
            if not tail_tokens:
                tail_tokens = [tail]
            
            leaves.append((relation, tail_tokens))
        
        if leaves:
            texts.append(code)
            leaves_list.append(leaves)
    
    return texts, leaves_list


def build_dataset_from_kg_full(
    triples_path: Path, 
    code_paths: List[Path], 
    max_seq_len: int = 128,
    max_samples: Optional[int] = None,
    max_leaves_per_sample: int = 5
) -> LeafyChainDatasetV2:
    """Build dataset using ALL triples from KG (not just 16-32).
    
    Args:
        triples_path: Path to KG triples file
        code_paths: List of code files to extract snippets from
        max_seq_len: Maximum sequence length
        max_samples: Optional limit on number of samples (for testing)
        max_leaves_per_sample: Max relation chains per sample
    
    Returns:
        LeafyChainDatasetV2 with all triples
    """
    print(f"Loading triples from {triples_path}")
    triples = load_all_triples(triples_path)
    print(f"Loaded {len(triples)} triples")
    
    # Group by head entity
    grouped = group_triples_by_head(triples)
    print(f"Grouped into {len(grouped)} unique head entities")
    
    # Load code snippets
    code_snippets = {}
    for code_path in code_paths:
        if code_path.exists():
            text = code_path.read_text(encoding="utf-8")
            # Use filename as key
            key = code_path.stem
            code_snippets[key] = text
            
            # Also try to extract function/class definitions
            # Simple heuristic: look for def/class statements
            import re
            for match in re.finditer(r'(?:def|class)\s+(\w+)', text):
                name = match.group(1)
                # Extract the definition block (rough heuristic)
                start = match.start()
                # Find end of block (next def/class or end of file)
                next_match = re.search(r'\n(?:def|class)\s+', text[start + len(match.group(0)):])
                if next_match:
                    end = start + len(match.group(0)) + next_match.start()
                else:
                    end = len(text)
                code_snippets[name] = text[start:end]
    
    print(f"Loaded {len(code_snippets)} code snippets")
    
    # Create training samples
    texts, leaves_list = create_training_samples(grouped, code_snippets, max_leaves_per_sample)
    
    if max_samples:
        texts = texts[:max_samples]
        leaves_list = leaves_list[:max_samples]
    
    print(f"Creating dataset with {len(texts)} samples")
    
    return LeafyChainDatasetV2(texts, leaves_list, max_seq_len=max_seq_len)


# Keep backward-compatible simple builder for quick testing
def build_dataset_from_kg_simple(
    triples_path: Path, 
    code_path: Path, 
    max_seq_len: int = 128, 
    limit: int = 1000,
    chunk_size: int = 5
) -> LeafyChainDatasetV2:
    """Simple builder that uses single code file (backward compatible).
    
    Args:
        triples_path: Path to triples file
        code_path: Single code file to use
        max_seq_len: Max sequence length
        limit: Max number of triples to use
        chunk_size: Number of leaves per sample
    """
    from .dataset import simple_tokenize
    
    text = code_path.read_text(encoding="utf-8")
    
    # Load triples
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
            # Split tail by dot
            toks = []
            for part in str(tail).split("."):
                toks.extend(simple_tokenize(part))
            leaves.append((rel, toks))
            count += 1
            if count >= limit:
                break
    
    # Chunk leaves
    groups = [leaves[i:i+chunk_size] for i in range(0, len(leaves), chunk_size)]
    if not groups:
        groups = [[]]
    
    texts = [text for _ in groups]
    
    return LeafyChainDatasetV2(texts, groups, max_seq_len=max_seq_len)
