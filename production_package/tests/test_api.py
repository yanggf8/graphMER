#!/usr/bin/env python3
"""
GraphMER-SE API Tests
Test suite for production inference server.
"""

import requests
import json
import time
import pytest

# Test configuration
BASE_URL = "http://localhost:8080"

# Sample code for testing
SAMPLE_CODES = {
    "python": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
    "java": """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""",
    "javascript": """
function factorial(n) {
    if (n === 0) return 1;
    return n * factorial(n - 1);
}
"""
}

def test_health_check():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("✅ Health check passed")

def test_info_endpoint():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_info" in data
    assert "endpoints" in data
    print("✅ Info endpoint passed")

def test_encode_endpoint():
    """Test code encoding endpoint"""
    for language, code in SAMPLE_CODES.items():
        response = requests.post(f"{BASE_URL}/encode", json={
            "code": code,
            "language": language
        })
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "processing_time_ms" in data
        assert len(data["embedding"]) > 0
        print(f"✅ Encoding test passed for {language}")

def test_batch_encode():
    """Test batch encoding endpoint"""
    codes = list(SAMPLE_CODES.values())
    response = requests.post(f"{BASE_URL}/batch_encode", json={
        "codes": codes,
        "batch_size": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == len(codes)
    print("✅ Batch encoding test passed")

def test_similarity_search():
    """Test similarity search endpoint"""
    query = SAMPLE_CODES["python"]
    candidates = list(SAMPLE_CODES.values())
    
    response = requests.post(f"{BASE_URL}/similarity", json={
        "query": query,
        "candidates": candidates,
        "top_k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    print("✅ Similarity search test passed")

def test_performance_benchmark():
    """Basic performance benchmark"""
    code = SAMPLE_CODES["python"]
    
    # Warm up
    requests.post(f"{BASE_URL}/encode", json={"code": code})
    
    # Measure performance
    start_time = time.time()
    num_requests = 10
    
    for _ in range(num_requests):
        response = requests.post(f"{BASE_URL}/encode", json={"code": code})
        assert response.status_code == 200
    
    total_time = time.time() - start_time
    avg_time = total_time / num_requests
    throughput = num_requests / total_time
    
    print(f"✅ Performance benchmark:")
    print(f"   Average latency: {avg_time*1000:.1f}ms")
    print(f"   Throughput: {throughput:.1f} req/s")

def test_error_handling():
    """Test error handling"""
    # Empty code
    response = requests.post(f"{BASE_URL}/encode", json={"code": ""})
    assert response.status_code == 500
    
    # Missing code field
    response = requests.post(f"{BASE_URL}/encode", json={})
    assert response.status_code == 400
    
    print("✅ Error handling test passed")

if __name__ == "__main__":
    print("🧪 Running GraphMER API Tests...")
    
    try:
        test_health_check()
        test_info_endpoint()
        test_encode_endpoint()
        test_batch_encode()
        test_similarity_search()
        test_performance_benchmark()
        test_error_handling()
        
        print("\n🎉 All tests passed! API is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the inference server is running at http://localhost:8080")