#!/usr/bin/env python3
"""
Benchmark CPU training speed to estimate time for 10k steps.
Tests actual training loop performance on current hardware.
"""
import time
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_v2 import KGDatasetV2
from src.training.tokenizer_bpe import BPETokenizer
from src.models.encoder import GraphAwareEncoder


def benchmark_training_speed(num_steps=10, config_path="configs/train_cpu.yaml"):
    """Run a mini training loop to measure steps/sec on CPU."""
    
    print("="*60)
    print("CPU Training Speed Benchmark")
    print("="*60)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"\n📋 Configuration:")
    print(f"  Device: {config['hardware']['device']}")
    print(f"  Batch size: {config['training_data']['micro_batch_size']}")
    print(f"  Gradient accumulation: {config['run']['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['training_data']['micro_batch_size'] * config['run']['gradient_accumulation_steps']}")
    print(f"  Max seq len: {config['training_data']['max_seq_len']}")
    print(f"  Model layers: {config['model']['num_layers']}")
    print(f"  Hidden size: {config['model']['hidden_size']}")
    
    # System info
    print(f"\n💻 System Info:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Available CPU cores: {torch.get_num_interop_threads()}")
    
    # Initialize tokenizer (dummy for speed test)
    print(f"\n🔧 Initializing components...")
    tokenizer = BPETokenizer()
    
    # Check if we have actual data
    kg_path = Path("data/kg/train.jsonl")
    if not kg_path.exists():
        print(f"  ⚠️  No training data found at {kg_path}")
        print(f"  Creating synthetic data for benchmark...")
        # Create minimal synthetic data
        kg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(kg_path, 'w') as f:
            for i in range(50):
                f.write(f'{{"head": "entity_{i}", "relation": "relates_to", "tail": "entity_{i+1}", "context": "def function_{i}(): pass"}}\n')
    
    # Load dataset
    dataset = KGDatasetV2(
        kg_file=str(kg_path),
        tokenizer=tokenizer,
        max_seq_len=config['training_data']['max_seq_len'],
        mask_prob=config['objectives']['mlm']['mask_prob']
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training_data']['micro_batch_size'],
        shuffle=True,
        num_workers=0,  # Single process for benchmark
        pin_memory=False
    )
    
    # Initialize model
    model = GraphAwareEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        intermediate_size=config['model']['intermediate_size'],
        dropout=config['model']['dropout'],
        max_position_embeddings=config['training_data']['max_seq_len'],
        use_rel_attention_bias=config['model']['use_rel_attention_bias']
    )
    
    device = torch.device(config['hardware']['device'])
    model = model.to(device)
    model.train()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    print(f"\n🏃 Running {num_steps} training steps...")
    print(f"  (This will take a few minutes on CPU...)")
    
    # Warmup step (compilation, cache warming)
    print(f"\n  Warmup step (not counted)...")
    try:
        batch = next(iter(dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Warmup complete")
    except Exception as e:
        print(f"  ⚠️  Warmup error: {e}")
        print(f"  Continuing with benchmark anyway...")
    
    # Actual benchmark
    step_times = []
    data_iter = iter(dataloader)
    
    print(f"\n  Benchmarking...")
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        step_start = time.time()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        if (step + 1) % 5 == 0:
            avg_time = sum(step_times[-5:]) / min(5, len(step_times))
            print(f"    Step {step+1}/{num_steps}: {step_time:.2f}s (avg: {avg_time:.2f}s/step)")
    
    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\n⏱️  Step Timing:")
    print(f"  Average: {avg_step_time:.2f} seconds/step")
    print(f"  Min:     {min_step_time:.2f} seconds/step")
    print(f"  Max:     {max_step_time:.2f} seconds/step")
    print(f"  Throughput: {1/avg_step_time:.2f} steps/second")
    
    # Calculate time estimates
    effective_batch = config['training_data']['micro_batch_size'] * config['run']['gradient_accumulation_steps']
    
    print(f"\n⏰ TIME ESTIMATES (with gradient_accumulation_steps={config['run']['gradient_accumulation_steps']}):")
    print(f"  Note: Each 'effective step' = {config['run']['gradient_accumulation_steps']} micro-batches")
    
    # Time for gradient accumulation
    time_per_effective_step = avg_step_time * config['run']['gradient_accumulation_steps']
    
    print(f"\n  Time per effective training step: {time_per_effective_step:.1f} seconds")
    print(f"  Time per effective training step: {time_per_effective_step/60:.2f} minutes")
    
    # Estimate for different step counts
    step_targets = [100, 500, 1000, 3500, 6500, 10000]
    print(f"\n  Training time estimates:")
    for steps in step_targets:
        total_seconds = steps * time_per_effective_step
        hours = total_seconds / 3600
        days = hours / 24
        
        if hours < 1:
            time_str = f"{total_seconds/60:.1f} minutes"
        elif hours < 24:
            time_str = f"{hours:.1f} hours"
        else:
            time_str = f"{days:.1f} days ({hours:.1f} hours)"
        
        print(f"    {steps:>5} steps: {time_str}")
    
    # Memory info
    if hasattr(torch.cuda, 'memory_allocated'):
        print(f"\n💾 Memory Usage:")
        print(f"  Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print(f"\n{'='*60}")
    print(f"✅ Benchmark complete!")
    print(f"{'='*60}\n")
    
    return {
        'avg_step_time': avg_step_time,
        'avg_effective_step_time': time_per_effective_step,
        'throughput': 1/avg_step_time,
        'config': config
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark CPU training speed')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to benchmark')
    parser.add_argument('--config', type=str, default='configs/train_cpu.yaml', help='Config file')
    args = parser.parse_args()
    
    results = benchmark_training_speed(num_steps=args.steps, config_path=args.config)
