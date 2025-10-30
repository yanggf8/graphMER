#!/bin/bash
# GraphMER-SE Ollama Installation Script

echo "🚀 Installing GraphMER-SE for Ollama..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Create the model
echo "📦 Creating GraphMER-SE model..."
ollama create graphmer-se -f Modelfile

if [ $? -eq 0 ]; then
    echo "✅ GraphMER-SE installed successfully!"
    echo ""
    echo "🎯 Quick test:"
    echo "   ollama run graphmer-se 'def hello(): return "world"'"
    echo ""
    echo "📚 See README.md for more examples"
else
    echo "❌ Installation failed. Check Modelfile and try again."
    exit 1
fi
