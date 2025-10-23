"""Fixed dataset implementation that properly uses BPE tokenizer vocabulary."""
from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from .tokenizer_bpe import create_code_tokenizer

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
    """Dataset that properly uses BPE tokenizer's vocabulary.
    
    Key differences from original:
    - Uses BPE tokenizer's vocab directly (8000 tokens)
    - No separate Vocab object that creates duplicate IDs
    - Relation tokens are encoded as regular text tokens
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
        
        # Build relation vocabulary (separate from token vocab)
        self.rel_stoi: Dict[str, int] = {"<none>": 0}
        
        self.samples: List[Dict] = []
        
        print(f"Building dataset with BPE tokenizer (vocab_size={self.vocab_size})")
        
        for text, leaves in zip(code_snippets, leaves_per_snip):
            sample = self._build_sample(text, leaves)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"Built {len(self.samples)} samples with vocab_size={self.vocab_size}")
    
    def _build_sample(self, text: str, leaves: List[Tuple[str, List[str]]]) -> Optional[Dict]:
        """Build a single training sample."""
        import random
        
        # Tokenize code with BPE
        code_token_ids = self.tokenizer.encode(text)
        
        # Build sequence: [CLS] code_tokens [SEP] [REL] rel tail_tokens [SEP] ...
        input_ids = [SPECIAL_TOKEN_IDS["[CLS]"]] + code_token_ids + [SPECIAL_TOKEN_IDS["[SEP]"]]
        rel_ids = [0] * len(input_ids)
        
        # Track where code ends for MLM masking
        code_end_idx = len(input_ids) - 1  # Before first [SEP]
        
        # Add leaves (relation chains)
        for rel, tail_tokens in leaves:
            # Register relation
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
        # Collect maskable positions first
        mlm_candidates = []
        for i in range(1, min(code_end_idx, len(input_ids))):
            token_id = input_ids[i]
            # Skip special tokens
            if token_id in SPECIAL_TOKEN_IDS.values() or token_id == REL_TOKEN_ID:
                continue
            mlm_candidates.append(i)
        
        # Mask tokens with probability mlm_prob, but ensure at least 1 is masked
        mlm_masked_positions = []
        for i in mlm_candidates:
            if random.random() < self.mlm_prob:
                mlm_masked_positions.append(i)
        
        # If nothing was masked, force mask one random token
        if not mlm_masked_positions and mlm_candidates:
            mlm_masked_positions = [random.choice(mlm_candidates)]
        
        # Apply masks
        for i in mlm_masked_positions:
            mlm_labels[i] = input_ids[i]
            input_ids[i] = SPECIAL_TOKEN_IDS["[MASK]"]
        
        # MNM: mask tail tokens in relation chains
        # Collect all tail token positions first
        mnm_candidates = []
        j = 0
        while j < len(input_ids):
            if input_ids[j] == REL_TOKEN_ID and rel_ids[j] > 0:
                # We found start of a relation chain
                current_rel_id = rel_ids[j]
                
                # Find all tokens with this rel_id (after [REL])
                for idx in range(j + 1, len(input_ids)):
                    if input_ids[idx] == SPECIAL_TOKEN_IDS["[SEP]"]:
                        if rel_ids[idx] == 0:  # End of this chain
                            j = idx
                            break
                    elif rel_ids[idx] == current_rel_id and input_ids[idx] not in SPECIAL_TOKEN_IDS.values():
                        # This is a tail token (part of relation chain)
                        mnm_candidates.append(idx)
            j += 1
        
        # Mask tokens with probability mnm_prob, but ensure at least 1 is masked
        mnm_masked_positions = []
        for idx in mnm_candidates:
            if random.random() < self.mnm_prob:
                mnm_masked_positions.append(idx)
        
        # If nothing was masked, force mask one random token
        if not mnm_masked_positions and mnm_candidates:
            mnm_masked_positions = [random.choice(mnm_candidates)]
        
        # Apply masks
        for idx in mnm_masked_positions:
            mnm_labels[idx] = input_ids[idx]
            input_ids[idx] = SPECIAL_TOKEN_IDS["[MASK]"]
        
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
