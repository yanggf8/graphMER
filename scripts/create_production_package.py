#!/usr/bin/env python3
"""
GraphMER-SE Production Package Creator
Creates a deployable package with model, tokenizer, and inference components.
"""

import shutil
import json
from pathlib import Path
import torch
import argparse

def create_production_package(model_path, output_dir="production_package"):
    """Create a complete production deployment package"""
    
    print("🚀 Creating GraphMER Production Package...")
    
    # Create directory structure
    package_dir = Path(output_dir)
    package_dir.mkdir(exist_ok=True)
    
    dirs = ["models", "api", "config", "tests", "docker", "monitoring"]
    for dir_name in dirs:
        (package_dir / dir_name).mkdir(exist_ok=True)
    
    # Copy model files
    print("📦 Copying model artifacts...")
    model_src = Path(model_path)
    model_dst = package_dir / "models" / "graphmer_model.pt"
    shutil.copy2(model_src, model_dst)
    
    # Copy supporting files
    supporting_files = [
        ("data/tokenizer/code_bpe.json", "models/tokenizer.json"),
        ("data/kg/seed_multilang.jsonl", "models/knowledge_graph.jsonl"),
        ("configs/train_v2_gpu.yaml", "config/model_config.yaml")
    ]
    
    for src, dst in supporting_files:
        src_path = Path(src)
        if src_path.exists():
            dst_path = package_dir / dst
            shutil.copy2(src_path, dst_path)
            print(f"  ✅ Copied {src} → {dst}")
        else:
            print(f"  ⚠️  Missing {src}")
    
    # Create model metadata
    print("📋 Creating model metadata...")
    metadata = {
        "model_name": "GraphMER-SE",
        "version": "1.0.0",
        "created": "2025-10-29",
        "parameters": "85M",
        "training_steps": 10000,
        "model_file": "models/graphmer_model.pt",
        "tokenizer_file": "models/tokenizer.json",
        "kg_file": "models/knowledge_graph.jsonl",
        "input_format": "code_text",
        "output_format": "embedding_vector",
        "supported_languages": ["python", "java", "javascript"],
        "requirements": [
            "torch>=2.0.0",
            "transformers>=4.20.0",
            "numpy>=1.21.0"
        ]
    }
    
    with open(package_dir / "models" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Model size analysis
    model_size_mb = model_dst.stat().st_size / (1024 * 1024)
    print(f"📊 Model size: {model_size_mb:.1f} MB")
    
    print(f"✅ Production package created at: {package_dir.absolute()}")
    print("\n🎯 Next steps:")
    print("  1. cd production_package")
    print("  2. python3 api/inference_server.py")
    print("  3. Test at http://localhost:8080/health")
    
    return package_dir

def main():
    parser = argparse.ArgumentParser(description="Create GraphMER production package")
    parser.add_argument("--model", default="logs/checkpoints/model_v2_20251029_051803_s42.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--output", default="production_package",
                       help="Output directory for production package")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    create_production_package(args.model, args.output)
    return True

if __name__ == "__main__":
    main()