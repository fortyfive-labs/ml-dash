"""
Generate optimizer_sweep.jsonl comparing SGD vs Adam

This script creates a sweep comparing SGD and Adam optimizers
across different learning rates.
"""
import json
from pathlib import Path


def generate_optimizer_sweep():
    """Generate optimizer comparison sweep configurations."""
    configs = [
        # SGD with different learning rates
        {"learning_rate": 0.1, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.001, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},

        # Adam with different learning rates
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "Adam", "momentum": 0.0},
        {"learning_rate": 0.001, "batch_size": 32, "optimizer": "Adam", "momentum": 0.0},
        {"learning_rate": 0.0001, "batch_size": 32, "optimizer": "Adam", "momentum": 0.0},
    ]

    # Write to optimizer_sweep.jsonl
    output_file = Path(__file__).parent / "optimizer_sweep.jsonl"
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"✓ Generated {len(configs)} optimizer configurations in {output_file.name}")
    return configs


if __name__ == "__main__":
    configs = generate_optimizer_sweep()

    print(f"\nPreview:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. opt={config['optimizer']:<5} lr={config['learning_rate']:<6}")
