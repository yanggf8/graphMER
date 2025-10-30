#!/usr/bin/env python3
"""
GraphMER-SE Ollama API Wrapper
Provides familiar GraphMER API interface using Ollama backend.
"""

import requests
import json
import numpy as np
from typing import List, Dict, Any

class OllamaGraphMER:
    """GraphMER API using Ollama backend"""
    
    def __init__(self, model_name="graphmer-se", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def encode_code(self, code_text: str, language: str = "python") -> np.ndarray:
        """Encode code using Ollama GraphMER model"""
        
        prompt = f"""Encode this {language} code into embedding vector:
```{language}
{code_text}
```
Return only the numeric embedding vector."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            # Parse embedding from response
            # This would need to be implemented based on actual model output
            return np.random.randn(768)  # Placeholder
            
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def similarity_search(self, query_code: str, candidate_codes: List[str], top_k: int = 10) -> List[Dict]:
        """Find similar code using Ollama GraphMER"""
        
        candidates_text = "\n".join([f"{i}: {code}" for i, code in enumerate(candidate_codes)])
        
        prompt = f"""Find the top {top_k} most similar code snippets to this query:

Query:
```
{query_code}
```

Candidates:
{candidates_text}

Return similarity scores and rankings."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Parse similarity results
            # Implementation would depend on model output format
            return [{"index": i, "similarity": 0.5, "code": code} 
                   for i, code in enumerate(candidate_codes[:top_k])]
            
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Ollama GraphMER is available"""
        try:
            # Check if model is available
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            
            model_available = any(model.get("name") == self.model_name for model in models)
            
            return {
                "status": "healthy" if model_available else "model_not_found",
                "model_name": self.model_name,
                "model_available": model_available,
                "ollama_running": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ollama_running": False
            }

# Example usage
if __name__ == "__main__":
    # Initialize GraphMER with Ollama
    graphmer = OllamaGraphMER()
    
    # Health check
    health = graphmer.health_check()
    print("Health:", health)
    
    if health["status"] == "healthy":
        # Test encoding
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        embedding = graphmer.encode_code(code, "python")
        print(f"Embedding shape: {embedding.shape}")
        
        # Test similarity
        candidates = [
            "def fib(x): return x if x <= 1 else fib(x-1) + fib(x-2)",
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "def add(a, b): return a + b"
        ]
        
        results = graphmer.similarity_search(code, candidates)
        print("Similarity results:", results)
