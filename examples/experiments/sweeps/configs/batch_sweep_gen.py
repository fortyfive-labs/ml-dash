"""
Generate batch_sweep.jsonl with batch size variations

This script creates a sweep testing different batch sizes
with fixed learning rate and optimizer.
"""
import json
from pathlib import Path


def generate_batch_sweep():
    """Generate batch size sweep configurations."""
    configs = [
        # Batch size variations with SGD
        {"learning_rate": 0.01, "batch_size": 16, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 64, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 128, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 256, "optimizer": "SGD", "momentum": 0.9},
    ]

    # Write to batch_sweep.jsonl
    output_file = Path(__file__).parent / "batch_sweep.jsonl"
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"✓ Generated {len(configs)} batch size configurations in {output_file.name}")
    return configs


if __name__ == "__main__":
    configs = generate_batch_sweep()

    print(f"\nPreview:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. batch={config['batch_size']:<3}")
