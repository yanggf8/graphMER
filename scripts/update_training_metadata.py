#!/usr/bin/env python3
import json
import csv

# Read final metrics
with open('logs/train_v2_metrics.csv') as f:
    rows = list(csv.DictReader(f))
    final = rows[-1]

# Update metadata
with open('ablation_metadata.json') as f:
    metadata = json.load(f)

metadata['training_results'] = {
    'steps_completed': 5000,
    'final_total_loss': float(final['total_loss']),
    'final_mlm_accuracy': float(final['mlm_validation_accuracy']),
    'final_mnm_accuracy': float(final['mnm_validation_accuracy']),
    'loss_reduction_percent': ((18.375 - 11.508) / 18.375) * 100
}

with open('ablation_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Updated ablation_metadata.json with training results")
