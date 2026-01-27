"""
Generate resnet_baseline.jsonl with standard ResNet configurations

This script creates baseline configurations for reproducible experiments.
Baselines use static paths (no datetime) for permanent reference.
"""
import json
from pathlib import Path


def generate_resnet_baseline():
    """Generate standard ResNet baseline configurations."""
    configs = [
        # ResNet-18 baseline
        {
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "SGD",
            "momentum": 0.9,
            "model": "ResNet-18",
            "epochs": 10,
        },

        # ResNet-50 baseline
        {
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "SGD",
            "momentum": 0.9,
            "model": "ResNet-50",
            "epochs": 10,
        },

        # ResNet with Adam
        {
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "Adam",
            "momentum": 0.0,
            "model": "ResNet-18",
            "epochs": 10,
        },
    ]

    # Write to resnet_baseline.jsonl
    output_file = Path(__file__).parent / "resnet_baseline.jsonl"
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"✓ Generated {len(configs)} baseline configurations in {output_file.name}")
    return configs


if __name__ == "__main__":
    configs = generate_resnet_baseline()

    print(f"\nBaseline Configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['model']:<12} opt={config['optimizer']:<5} lr={config['learning_rate']:<6} batch={config['batch_size']}")
