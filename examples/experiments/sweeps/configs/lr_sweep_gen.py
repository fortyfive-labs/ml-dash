"""
Generate lr_sweep.jsonl with learning rate variations

This script creates a focused sweep testing different learning rates
with SGD optimizer and fixed batch size.
"""
import json
from pathlib import Path


def generate_lr_sweep():
    """Generate learning rate sweep configurations."""
    configs = [
        # Learning rate variations with SGD
        {"learning_rate": 0.1, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.05, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.005, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.001, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
    ]

    # Write to lr_sweep.jsonl
    output_file = Path(__file__).parent / "lr_sweep.jsonl"
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"✓ Generated {len(configs)} learning rate configurations in {output_file.name}")
    return configs


if __name__ == "__main__":
    configs = generate_lr_sweep()

    print(f"\nPreview:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. lr={config['learning_rate']:<6}")
