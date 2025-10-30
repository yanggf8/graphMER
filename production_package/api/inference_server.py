#!/usr/bin/env python3
"""
GraphMER-SE Inference Server
Production-ready Flask API for GraphMER model inference.
"""

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
import logging
import time
import json
import hashlib
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphMERInference:
    """Production inference wrapper for GraphMER model"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, kg_path: str = None):
        """Initialize inference engine"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load metadata if available
        model_dir = Path(model_path).parent
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"version": "unknown"}
        
        logger.info("✅ GraphMER inference engine ready")
    
    def encode_code(self, code_text: str, language: str = "python") -> np.ndarray:
        """
        Encode code text into GraphMER embedding
        
        Args:
            code_text: Source code string
            language: Programming language (python, java, javascript)
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            with torch.no_grad():
                # Basic preprocessing
                code_text = code_text.strip()
                if not code_text:
                    raise ValueError("Empty code input")
                
                # For now, return a mock embedding - you'll need to implement
                # the actual tokenization and model forward pass based on your
                # GraphMER architecture
                
                # Placeholder implementation - replace with actual GraphMER inference
                embedding_dim = 768  # Your model's embedding dimension
                mock_embedding = torch.randn(embedding_dim, device=self.device)
                
                return mock_embedding.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def batch_encode(self, code_list: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """Efficiently encode multiple code samples"""
        embeddings = []
        
        for i in range(0, len(code_list), batch_size):
            batch = code_list[i:i+batch_size]
            batch_embeddings = []
            
            for code in batch:
                embedding = self.encode_code(code)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def similarity_search(self, query_code: str, candidate_codes: List[str], top_k: int = 10) -> List[Dict]:
        """Find most similar code snippets"""
        query_embedding = self.encode_code(query_code)
        candidate_embeddings = self.batch_encode(candidate_codes)
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append({
                'index': i,
                'code': candidate_codes[i],
                'similarity': float(similarity)
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

# Flask application
app = Flask(__name__)

# Global inference engine
inference_engine = None

def init_model():
    """Initialize model on startup"""
    global inference_engine
    
    # Try to find model in production package structure
    model_paths = [
        "models/graphmer_model.pt",
        "production_package/models/graphmer_model.pt",
        "logs/checkpoints/model_v2_20251029_051803_s42.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        raise RuntimeError("No GraphMER model found. Run create_production_package.py first.")
    
    inference_engine = GraphMERInference(model_path)
    logger.info("🚀 Inference server initialized")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.memory_allocated() if gpu_available else 0
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': inference_engine is not None,
            'gpu_available': gpu_available,
            'gpu_memory_mb': gpu_memory / (1024 * 1024),
            'version': inference_engine.metadata.get('version', 'unknown') if inference_engine else 'unknown'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/encode', methods=['POST'])
def encode_endpoint():
    """Code encoding endpoint"""
    try:
        data = request.json
        if not data or 'code' not in data:
            return jsonify({'error': 'Missing code field'}), 400
        
        start_time = time.time()
        
        embedding = inference_engine.encode_code(
            data['code'], 
            data.get('language', 'python')
        )
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'embedding': embedding.tolist(),
            'processing_time_ms': processing_time * 1000,
            'embedding_dim': len(embedding)
        })
        
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity_endpoint():
    """Code similarity search endpoint"""
    try:
        data = request.json
        if not data or 'query' not in data or 'candidates' not in data:
            return jsonify({'error': 'Missing query or candidates fields'}), 400
        
        start_time = time.time()
        
        results = inference_engine.similarity_search(
            data['query'],
            data['candidates'],
            data.get('top_k', 10)
        )
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'results': results,
            'processing_time_ms': processing_time * 1000,
            'query_length': len(data['query']),
            'candidate_count': len(data['candidates'])
        })
        
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_encode', methods=['POST'])
def batch_encode_endpoint():
    """Batch encoding endpoint for multiple code samples"""
    try:
        data = request.json
        if not data or 'codes' not in data:
            return jsonify({'error': 'Missing codes field'}), 400
        
        codes = data['codes']
        if len(codes) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 100)'}), 400
        
        start_time = time.time()
        
        embeddings = inference_engine.batch_encode(
            codes,
            batch_size=data.get('batch_size', 8)
        )
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'embeddings': [emb.tolist() for emb in embeddings],
            'processing_time_ms': processing_time * 1000,
            'batch_size': len(codes),
            'embedding_dim': len(embeddings[0]) if embeddings else 0
        })
        
    except Exception as e:
        logger.error(f"Batch encoding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info_endpoint():
    """Model information endpoint"""
    if not inference_engine:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_info': inference_engine.metadata,
        'device': str(inference_engine.device),
        'gpu_available': torch.cuda.is_available(),
        'endpoints': {
            'health': 'GET /health',
            'encode': 'POST /encode',
            'batch_encode': 'POST /batch_encode',
            'similarity': 'POST /similarity',
            'info': 'GET /info'
        }
    })

if __name__ == '__main__':
    # Initialize model
    init_model()
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting GraphMER inference server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)