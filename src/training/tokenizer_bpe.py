"""Production-ready BPE tokenizer for GraphMER-SE using HuggingFace tokenizers library."""
from typing import List, Optional, Union
from pathlib import Path
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


class CodeBPETokenizer:
    """Real BPE tokenizer for code with proper subword tokenization."""
    
    # Special tokens for GraphMER-SE
    SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[REL]", "[UNK]"]
    
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 30000):
        """
        Initialize BPE tokenizer.
        
        Args:
            tokenizer_path: Path to saved tokenizer. If None, creates new untrained tokenizer.
            vocab_size: Vocabulary size for training (default: 30k)
        """
        self.vocab_size = vocab_size
        
        if tokenizer_path and Path(tokenizer_path).exists():
            # Load existing tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Create new BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            
            # Use ByteLevel pre-tokenizer for better code handling
            # This splits on whitespace and punctuation while preserving structure
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            
            # Post-processor for special tokens
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    def train(self, files: List[str], output_path: Optional[str] = None):
        """
        Train BPE tokenizer on corpus files.
        
        Args:
            files: List of file paths to train on
            output_path: Optional path to save trained tokenizer
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            min_frequency=2,  # Minimum frequency for token to be included
        )
        
        print(f"Training BPE tokenizer on {len(files)} files...")
        self.tokenizer.train(files, trainer)
        print(f"✓ Training complete. Vocab size: {self.tokenizer.get_vocab_size()}")
        
        if output_path:
            self.save(output_path)
    
    def train_from_iterator(self, iterator, output_path: Optional[str] = None):
        """
        Train BPE tokenizer from text iterator.
        
        Args:
            iterator: Iterator yielding text strings
            output_path: Optional path to save trained tokenizer
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            min_frequency=2,
        )
        
        print(f"Training BPE tokenizer from iterator...")
        self.tokenizer.train_from_iterator(iterator, trainer)
        print(f"✓ Training complete. Vocab size: {self.tokenizer.get_vocab_size()}")
        
        if output_path:
            self.save(output_path)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs using BPE.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encode multiple texts in batch."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]
    
    def get_vocab(self) -> dict:
        """Get vocabulary dictionary (token -> id)."""
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token string to ID."""
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert token ID to string."""
        return self.tokenizer.id_to_token(token_id)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        print(f"✓ Tokenizer saved to {path}")
    
    def __len__(self):
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def get_special_token_ids(self) -> dict:
        """Get special token IDs."""
        return {
            token: self.token_to_id(token)
            for token in self.SPECIAL_TOKENS
        }


def train_code_tokenizer(
    corpus_paths: List[str],
    output_path: str = "data/tokenizer/code_bpe.json",
    vocab_size: int = 30000
) -> CodeBPETokenizer:
    """
    Train a BPE tokenizer on code corpus.
    
    Args:
        corpus_paths: List of file paths or glob patterns
        output_path: Path to save trained tokenizer
        vocab_size: Vocabulary size
        
    Returns:
        Trained tokenizer
    """
    import glob
    
    # Expand glob patterns
    all_files = []
    for pattern in corpus_paths:
        if '*' in pattern:
            all_files.extend(glob.glob(pattern, recursive=True))
        else:
            all_files.append(pattern)
    
    # Filter to existing files
    all_files = [f for f in all_files if Path(f).exists()]
    
    print(f"Found {len(all_files)} files for training")
    
    # Create and train tokenizer
    tokenizer = CodeBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_files, output_path=output_path)
    
    return tokenizer


def create_code_tokenizer(tokenizer_path: str = "data/tokenizer/code_bpe_large.json") -> CodeBPETokenizer:
    """
    Load or create code tokenizer.
    
    Args:
        tokenizer_path: Path to tokenizer file (default: code_bpe_large.json with 8K vocab)
        
    Returns:
        CodeBPETokenizer instance
    """
    if Path(tokenizer_path).exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        return CodeBPETokenizer(tokenizer_path=tokenizer_path)
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}")
        # Try fallback to standard BPE
        fallback_path = "data/tokenizer/code_bpe.json"
        if Path(fallback_path).exists():
            print(f"Using fallback tokenizer: {fallback_path}")
            return CodeBPETokenizer(tokenizer_path=fallback_path)
        # Auto-train a minimal tokenizer from available corpus
        try:
            print("Auto-training a minimal tokenizer from data/raw samples...")
            corpus_patterns = [
                "data/raw/python_samples/**/*.py",
                "data/raw/java_samples/**/*.java",
            ]
            tok = train_code_tokenizer(corpus_patterns, output_path=fallback_path, vocab_size=8000)
            return tok
        except Exception as e:
            print(f"Auto-train failed: {e}. Creating untrained tokenizer. Call .train() before use.")
            return CodeBPETokenizer()


if __name__ == "__main__":
    # Training script
    import sys
    
    print("=" * 60)
    print("Training BPE Tokenizer for GraphMER-SE")
    print("=" * 60)
    
    # Corpus paths
    corpus_patterns = [
        "data/raw/python_samples/**/*.py",
        "data/raw/java_samples/**/*.java",
    ]
    
    # Train tokenizer
    tokenizer = train_code_tokenizer(
        corpus_paths=corpus_patterns,
        output_path="data/tokenizer/code_bpe.json",
        vocab_size=30000
    )
    
    # Test tokenizer
    print("\n" + "=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)
    
    test_cases = [
        "def camelCaseFunction(): return True",
        "getUserById",
        "MyClass.methodName",
        "import numpy as np",
        "private static final String API_KEY = \"test123\";",
    ]
    
    for text in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\nInput:   {text}")
        print(f"Tokens:  {len(tokens)} tokens")
        print(f"IDs:     {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Decoded: {decoded}")
    
    # Print vocab stats
    print("\n" + "=" * 60)
    print("Vocabulary Statistics")
    print("=" * 60)
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.get_special_token_ids()}")
    
    print("\n✓ Tokenizer training complete!")
