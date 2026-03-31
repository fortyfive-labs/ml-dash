---
sidebar_label: Experiment Comparison
---

# Experiment Comparison

Track multiple experiments and compare results across architectures.

```python
"""Compare multiple experiments."""
from ml_dash import Experiment
import random

def train_model(architecture, experiment):
    base_acc = {"cnn": 0.85, "resnet": 0.90, "vit": 0.92}[architecture]
    epochs = 20
    for epoch in range(epochs):
        acc = min(base_acc, 0.5 + epoch * (base_acc - 0.5) / epochs + random.uniform(-0.02, 0.02))
        experiment.metrics("train").log(accuracy=acc, epoch=epoch)
    return acc

def compare_architectures():
    architectures = ["cnn", "resnet", "vit"]
    results = {}

    for arch in architectures:
        with Experiment(
            prefix=f"alice/architecture-comparison/comparison-{arch}",
            description=f"Training {arch} on CIFAR-10",
            tags=["comparison", arch, "cifar10"]
        ).run as experiment:
            experiment.params.set(architecture=arch, dataset="cifar10", batch_size=128, epochs=20)
            experiment.log(f"Training {arch} architecture")

            final_acc = train_model(arch, experiment)
            experiment.log(f"Final accuracy: {final_acc:.4f}")
            results[arch] = final_acc

    for arch in sorted(results, key=results.get, reverse=True):
        print(f"{arch:10s}: {results[arch]:.4f}")

if __name__ == "__main__":
    compare_architectures()
```
