"""Code-aware BPE tokenizer for GraphMER-SE."""
from typing import List, Dict
import re
from pathlib import Path


class CodeBPETokenizer:
    """Minimal BPE tokenizer for code with identifier preservation."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[tuple] = []
        self._build_base_vocab()
    
    def _build_base_vocab(self):
        """Initialize with special tokens and common code patterns."""
        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[REL]", "[UNK]"]
        
        # Add special tokens
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
        
        # Add single characters (ASCII printable)
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into word-level tokens preserving code structure."""
        # Preserve identifiers, strings, numbers
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|"[^"]*"|\'[^\']*\'|\d+\.?\d*|[^\w\s])'
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t.strip()]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = self._get_word_tokens(text)
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                # Fallback to character-level for OOV
                for char in word:
                    token_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(tid, "[UNK]") for tid in token_ids]
        return "".join(tokens)
    
    def __len__(self):
        return len(self.vocab)


def create_code_tokenizer(corpus_files: List[Path] = None) -> CodeBPETokenizer:
    """Factory function to create tokenizer."""
    tokenizer = CodeBPETokenizer()
    
    # TODO: Train on corpus if provided
    if corpus_files:
        # For now, use the base tokenizer
        # Future: implement BPE training on corpus
        pass
    
    return tokenizer
