"""Tests for BPE tokenizer."""
import pytest
from pathlib import Path
from src.training.tokenizer_bpe import CodeBPETokenizer, create_code_tokenizer


class TestBPETokenizer:
    """Test suite for BPE tokenizer."""
    
    def test_tokenizer_loads(self):
        """Test that trained tokenizer can be loaded."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        assert tokenizer is not None
        assert len(tokenizer) > 1000  # Should have substantial vocab
    
    def test_special_tokens(self):
        """Test that special tokens are present."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        special_tokens = tokenizer.get_special_token_ids()
        
        assert "[PAD]" in special_tokens
        assert "[CLS]" in special_tokens
        assert "[SEP]" in special_tokens
        assert "[MASK]" in special_tokens
        assert "[REL]" in special_tokens
        assert "[UNK]" in special_tokens
    
    def test_encode_decode(self):
        """Test that encoding and decoding works."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        text = "def my_function(): return True"
        
        # Encode
        token_ids = tokenizer.encode(text)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
        # Decoded text should be similar (BPE may add spaces)
        assert "def" in decoded
        assert "function" in decoded
    
    def test_camelcase_splitting(self):
        """Test that camelCase identifiers are split into subwords."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        text = "getUserById"
        
        token_ids = tokenizer.encode(text)
        # Should split into multiple tokens (not just 1)
        assert len(token_ids) > 1
        assert len(token_ids) < 15  # But not too many
    
    def test_vocab_bounded(self):
        """Test that vocabulary is bounded."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        vocab_size = len(tokenizer)
        
        # Should be in reasonable range (not unbounded)
        assert 1000 < vocab_size < 50000
    
    def test_batch_encoding(self):
        """Test batch encoding."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        texts = [
            "def function1(): pass",
            "class MyClass: pass",
            "import numpy as np"
        ]
        
        batch_ids = tokenizer.encode_batch(texts)
        assert len(batch_ids) == 3
        assert all(isinstance(ids, list) for ids in batch_ids)
        assert all(len(ids) > 0 for ids in batch_ids)
    
    def test_token_to_id_mapping(self):
        """Test token to ID and ID to token mapping."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        
        # Test special tokens
        pad_id = tokenizer.token_to_id("[PAD]")
        assert pad_id is not None
        assert tokenizer.id_to_token(pad_id) == "[PAD]"
        
        cls_id = tokenizer.token_to_id("[CLS]")
        assert cls_id is not None
        assert tokenizer.id_to_token(cls_id) == "[CLS]"
    
    def test_python_code_tokenization(self):
        """Test tokenization of Python code."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        code = "def process_user_data(user_id): return User.objects.get(id=user_id)"
        
        token_ids = tokenizer.encode(code)
        assert len(token_ids) > 5  # Should have multiple tokens
        
        # Decode should preserve semantic meaning
        decoded = tokenizer.decode(token_ids)
        assert "def" in decoded
        assert "user" in decoded.lower()
    
    def test_java_code_tokenization(self):
        """Test tokenization of Java code."""
        tokenizer = create_code_tokenizer("data/tokenizer/code_bpe.json")
        code = "public class UserService { private String apiKey; }"
        
        token_ids = tokenizer.encode(code)
        assert len(token_ids) > 5  # Should have multiple tokens
        
        decoded = tokenizer.decode(token_ids)
        assert "public" in decoded
        assert "class" in decoded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
