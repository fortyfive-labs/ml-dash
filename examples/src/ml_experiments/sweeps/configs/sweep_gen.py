"""
Generate sweep.jsonl with hyperparameter configurations

This script generates the main sweep file testing:
- Learning rate variations
- Batch size variations
- Optimizer comparisons (SGD vs Adam)
"""
import json
from pathlib import Path


def generate_sweep():
    """Generate main hyperparameter sweep configurations."""
    configs = [
        # Learning rate sweep with SGD
        {"learning_rate": 0.1, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.001, "batch_size": 32, "optimizer": "SGD", "momentum": 0.9},

        # Batch size sweep with SGD
        {"learning_rate": 0.01, "batch_size": 64, "optimizer": "SGD", "momentum": 0.9},
        {"learning_rate": 0.01, "batch_size": 128, "optimizer": "SGD", "momentum": 0.9},

        # Adam optimizer variations
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "Adam", "momentum": 0.0},
        {"learning_rate": 0.001, "batch_size": 64, "optimizer": "Adam", "momentum": 0.0},
        {"learning_rate": 0.0001, "batch_size": 128, "optimizer": "Adam", "momentum": 0.0},
    ]

    # Write to sweep.jsonl
    output_file = Path(__file__).parent / "sweep.jsonl"
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"✓ Generated {len(configs)} configurations in {output_file.name}")
    return configs


if __name__ == "__main__":
    configs = generate_sweep()

    print(f"\nPreview:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. lr={config['learning_rate']:<6} batch={config['batch_size']:<3} opt={config['optimizer']}")
