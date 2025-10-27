"""Fixed dataset implementation that properly uses BPE tokenizer vocabulary with Leafy Chain encoding."""
from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from .tokenizer_bpe import create_code_tokenizer
from ..encoding.leafy_chain import LeafyChainEncoder

# Special token IDs - must match BPE tokenizer configuration
SPECIAL_TOKEN_IDS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
    "[UNK]": 4,
}

# Additional special tokens for our task
REL_TOKEN = "[REL]"
REL_TOKEN_ID = 5  # Reserved ID for relation marker


class LeafyChainDatasetV2(Dataset):
    """Dataset that properly uses BPE tokenizer's vocabulary with Leafy Chain encoding.
    
    Key differences from original:
    - Uses BPE tokenizer's vocab directly (8000 tokens)
    - Uses Leafy Chain Graph Encoding for KG triples
    - Preserves graph structure in linearized sequences
    """
    
    def __init__(
        self, 
        code_snippets: List[str], 
        leaves_per_snip: List[List[Tuple[str, List[str]]]], 
        max_seq_len: int = 128, 
        mlm_prob: float = 0.15, 
        mnm_prob: float = 0.2
    ):
        self.tokenizer = create_code_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.max_seq_len = max_seq_len
        self.mlm_prob = mlm_prob
        self.mnm_prob = mnm_prob
        
        # Initialize Leafy Chain encoder
        self.leafy_encoder = LeafyChainEncoder()
        
        # Build relation vocabulary (separate from token vocab)
        self.rel_stoi: Dict[str, int] = {"<none>": 0}
        
        self.samples: List[Dict] = []
        
        print(f"Building dataset with BPE tokenizer + Leafy Chain encoding (vocab_size={self.vocab_size})")
        
        for text, leaves in zip(code_snippets, leaves_per_snip):
            sample = self._build_sample(text, leaves)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"Built {len(self.samples)} samples with vocab_size={self.vocab_size}")
    
    def _build_sample(self, text: str, leaves: List[Tuple[str, List[str]]]) -> Optional[Dict]:
        """Build a single training sample using Leafy Chain encoding."""
        import random
        
        # Tokenize code with BPE
        code_token_ids = self.tokenizer.encode(text)
        
        # Convert leaves to triples for Leafy Chain encoding
        triples = []
        for rel, tail_tokens in leaves:
            # Create pseudo-triples from relation-tail pairs
            # Use first tail token as entity, rest as attributes
            if tail_tokens:
                head_entity = "code_context"  # Placeholder head
                tail_entity = tail_tokens[0] if tail_tokens else "unknown"
                triples.append((head_entity, rel, tail_entity))
        
        # Apply Leafy Chain encoding to triples
        if triples:
            chain_tokens = self.leafy_encoder.linearize_graph(triples)
            
            # Build relation vocabulary from chain tokens
            for token in chain_tokens:
                if token.startswith("[REL]"):
                    rel_name = token[5:]  # Remove "[REL]" prefix
                    if rel_name not in self.rel_stoi:
                        self.rel_stoi[rel_name] = len(self.rel_stoi)
        else:
            chain_tokens = []
        
        # Build sequence: [CLS] code_tokens [SEP] chain_tokens [SEP]
        input_ids = [SPECIAL_TOKEN_IDS["[CLS]"]] + code_token_ids + [SPECIAL_TOKEN_IDS["[SEP]"]]
        rel_ids = [0] * len(input_ids)
        
        # Track where code ends for MLM masking
        code_end_idx = len(input_ids) - 1  # Before first [SEP]
        
        # Add Leafy Chain encoded tokens
        if chain_tokens:
            # Get relation IDs for chain tokens
            chain_rel_ids = self.leafy_encoder.get_relation_ids(chain_tokens, self.rel_stoi)
            
            # Encode chain tokens
            for token, rel_id in zip(chain_tokens, chain_rel_ids):
                if token.startswith("[REL]") or token.startswith("[ENT]"):
                    # Encode special tokens as text
                    token_ids = self.tokenizer.encode(token)
                    input_ids.extend(token_ids)
                    rel_ids.extend([rel_id] * len(token_ids))
                elif token == "[CHAIN]":
                    # Chain separator
                    sep_ids = self.tokenizer.encode(token)
                    input_ids.extend(sep_ids)
                    rel_ids.extend([0] * len(sep_ids))
                else:
                    # Regular token
                    token_ids = self.tokenizer.encode(token)
                    input_ids.extend(token_ids)
                    rel_ids.extend([rel_id] * len(token_ids))
            
            # Add final separator
            input_ids.append(SPECIAL_TOKEN_IDS["[SEP]"])
            rel_ids.append(0)
            if rel not in self.rel_stoi:
                self.rel_stoi[rel] = len(self.rel_stoi)
            r_id = self.rel_stoi[rel]
            
            # Encode [REL] marker - use reserved ID
            input_ids.append(REL_TOKEN_ID)
            rel_ids.append(r_id)
            
            # Encode relation name as text
            rel_token_ids = self.tokenizer.encode(rel)
            input_ids.extend(rel_token_ids)
            rel_ids.extend([r_id] * len(rel_token_ids))
            
            # Encode tail tokens
            tail_text = " ".join(tail_tokens)
            tail_token_ids = self.tokenizer.encode(tail_text)
            input_ids.extend(tail_token_ids)
            rel_ids.extend([r_id] * len(tail_token_ids))
            
            # Add separator
            input_ids.append(SPECIAL_TOKEN_IDS["[SEP]"])
            rel_ids.append(0)
        
        # Truncate to max length
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            rel_ids = rel_ids[:self.max_seq_len]
        
        # Create labels for MLM and MNM
        mlm_labels = [-100] * len(input_ids)
        mnm_labels = [-100] * len(input_ids)
        
        # MLM: mask some code tokens (excluding special tokens)
        mlm_candidates = []
        for i in range(1, min(code_end_idx, len(input_ids))):
            token_id = input_ids[i]
            if token_id not in SPECIAL_TOKEN_IDS.values():
                mlm_candidates.append(i)
        
        # Mask tokens with probability mlm_prob
        mlm_masked_positions = []
        for i in mlm_candidates:
            if random.random() < self.mlm_prob:
                mlm_masked_positions.append(i)
        
        # Ensure at least 1 token is masked
        if not mlm_masked_positions and mlm_candidates:
            mlm_masked_positions = [random.choice(mlm_candidates)]
        
        # Apply MLM masks
        for i in mlm_masked_positions:
            mlm_labels[i] = input_ids[i]
            input_ids[i] = SPECIAL_TOKEN_IDS["[MASK]"]
        
        # MNM: mask relation/entity tokens
        mnm_candidates = []
        for i in range(code_end_idx + 1, len(input_ids)):
            if rel_ids[i] > 0 and input_ids[i] not in SPECIAL_TOKEN_IDS.values():
                mnm_candidates.append(i)
        
        # Mask MNM tokens
        mnm_masked_positions = []
        for i in mnm_candidates:
            if random.random() < self.mnm_prob:
                mnm_masked_positions.append(i)
        
        # Ensure minimum MNM masks
        min_mnm = min(3, len(mnm_candidates))
        while len(mnm_masked_positions) < min_mnm and len(mnm_masked_positions) < len(mnm_candidates):
            remaining = [i for i in mnm_candidates if i not in mnm_masked_positions]
            if remaining:
                mnm_masked_positions.append(random.choice(remaining))
        
        # Apply MNM masks
        for i in mnm_masked_positions:
            mnm_labels[i] = input_ids[i]
            input_ids[i] = SPECIAL_TOKEN_IDS["[MASK]"]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "mnm_labels": mnm_labels,
            "rel_ids": rel_ids,
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["input_ids"], dtype=torch.long),
            torch.tensor(s["attention_mask"], dtype=torch.long),
            torch.tensor(s["mlm_labels"], dtype=torch.long),
            torch.tensor(s["mnm_labels"], dtype=torch.long),
            torch.tensor(s["rel_ids"], dtype=torch.long),
        )
