# GraphMER-SE for Ollama

## Quick Start

```bash
# Import the model
ollama create graphmer-se -f Modelfile

# Run the model
ollama run graphmer-se

# Example: Encode Python code
ollama run graphmer-se "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"

# Example: Compare code snippets
ollama run graphmer-se "Compare similarity: def add(a,b): return a+b vs def sum(x,y): return x+y"
```

## API Usage

```bash
# Start Ollama server
ollama serve

# Encode code via API
curl http://localhost:11434/api/generate \
  -d '{
    "model": "graphmer-se",
    "prompt": "def hello(): return \"world\"",
    "stream": false
  }'
```

## Model Information

- **Architecture**: GraphMER (Graph-based Transformer)
- **Parameters**: 85M
- **Training**: 10,000 steps on multilingual code
- **Languages**: Python, Java, JavaScript
- **Capabilities**: Code encoding, similarity search, graph reasoning

## Features

✅ Multi-language code understanding  
✅ Graph-structured code relationships  
✅ High-quality embeddings (768D)  
✅ Fast inference (optimized for local deployment)  
✅ No internet required (runs locally)  

## Performance

- **Model Size**: ~350MB (quantized)
- **Memory**: 2-4GB RAM
- **Speed**: <100ms per request
- **Hardware**: CPU or GPU (CUDA optional)
