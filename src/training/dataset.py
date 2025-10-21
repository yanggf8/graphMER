from __future__ import annotations
import re
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
    "[REL]": 4,
}


def simple_tokenize(text: str) -> List[str]:
    # Split on non-alphanumeric except underscore and dot
    return [t for t in re.split(r"([^A-Za-z0-9_\.])", text) if t and not t.isspace()]


class Vocab:
    def __init__(self):
        self.stoi: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.itos: List[str] = [None] * len(SPECIAL_TOKENS)
        for k, v in SPECIAL_TOKENS.items():
            self.itos[v] = k

    def add(self, tok: str) -> int:
        if tok not in self.stoi:
            idx = len(self.stoi)
            self.stoi[tok] = idx
            self.itos.append(tok)
        return self.stoi[tok]

    def encode(self, toks: List[str]) -> List[int]:
        return [self.add(t) for t in toks]

    def __len__(self):
        return len(self.stoi)


class LeafyChainDataset(Dataset):
    def __init__(self, code_snippets: List[str], leaves_per_snip: List[List[Tuple[str, List[str]]]], max_seq_len: int = 128, mlm_prob: float = 0.15, mnm_prob: float = 0.2):
        self.vocab = Vocab()
        self.rel_stoi: Dict[str, int] = {"<none>": 0}
        self.samples: List[Dict] = []
        for text, leaves in zip(code_snippets, leaves_per_snip):
            code_tokens = ["[CLS]"] + simple_tokenize(text) + ["[SEP]"]
            seq: List[str] = list(code_tokens)
            rel_ids: List[int] = [0] * len(seq)
            # attach leaves as REL, relation, tail tokens, SEP
            for rel, tail_tokens in leaves:
                if rel not in self.rel_stoi:
                    self.rel_stoi[rel] = len(self.rel_stoi)
                r_id = self.rel_stoi[rel]
                seq += ["[REL]", rel] + tail_tokens + ["[SEP]"]
                rel_ids += [r_id] * (2 + len(tail_tokens)) + [0]
            # truncate
            seq = seq[:max_seq_len]
            rel_ids = rel_ids[:max_seq_len]
            input_ids = self.vocab.encode(seq)
            # labels for MLM (only on code part) and MNM (on tail tokens after [REL])
            mlm_labels = [-100] * len(input_ids)
            mnm_labels = [-100] * len(input_ids)
            # MLM: mask some code tokens (excluding specials)
            for i, tok in enumerate(seq):
                if tok in SPECIAL_TOKENS or tok in ("[REL]",):
                    continue
                if tok == "[SEP]":
                    # allow masking code seg upto first sep only
                    break
                if (i % 7) == 0 and tok not in ("[CLS]",):
                    mlm_labels[i] = input_ids[i]
                    input_ids[i] = SPECIAL_TOKENS["[MASK]"]
            # MNM: mask leaf tail tokens (after [REL] until [SEP])
            j = 0
            while j < len(seq):
                if seq[j] == "[REL]":
                    k = j + 2
                    while k < len(seq) and seq[k] != "[SEP]":
                        mnm_labels[k] = input_ids[k]
                        input_ids[k] = SPECIAL_TOKENS["[MASK]"]
                        k += 1
                    j = k
                else:
                    j += 1
            attn = [1] * len(input_ids)
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attn,
                "mlm_labels": mlm_labels,
                "mnm_labels": mnm_labels,
                "rel_ids": rel_ids,
            })

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
