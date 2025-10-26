#!/usr/bin/env python3
"""Training script with MNM debugging."""
import sys
from pathlib import Path
import argparse
import yaml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--mnm_weight", type=float, default=1.0, help="MNM loss weight")
    parser.add_argument("--log_mnm_debug", type=int, default=0, help="Log MNM debug every N steps")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    
    # Quick MNM debug run
    from src.training.dataset_v2 import LeafyChainDatasetV2
    from src.training.kg_dataset_builder_v2 import build_dataset_from_kg_simple
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.metrics import masked_token_accuracy
    
    # Build small dataset
    kg_path = Path("data/kg/enhanced_multilang.jsonl")
    code_snippets, leaves_per_snip = build_dataset_from_kg_simple(kg_path, max_samples=100)
    dataset = LeafyChainDatasetV2(code_snippets, leaves_per_snip, max_seq_len=128)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    
    # Check a few samples for MNM labels
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        mnm_labels = sample[3]  # vmn
        valid_positions = (mnm_labels != -100).sum().item()
        unique_labels = torch.unique(mnm_labels[mnm_labels != -100])
        print(f"Sample {i}: MNM valid positions: {valid_positions}/{len(mnm_labels)}")
        print(f"  Unique MNM labels: {unique_labels.tolist()}")
        
        if valid_positions == 0:
            print("  🚨 No valid MNM labels in this sample!")

if __name__ == "__main__":
    main()
