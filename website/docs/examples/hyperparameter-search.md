---
sidebar_label: Hyperparameter Search
---

# Hyperparameter Search

Grid search over hyperparameters with full tracking.

```python
"""Hyperparameter search example."""
from itertools import product
import random
from ml_dash import Experiment

def train_with_config(lr, batch_size, experiment):
    for epoch in range(10):
        loss = 1.0 / (epoch + 1) * (lr / 0.01) + random.uniform(-0.05, 0.05)
        accuracy = min(0.95, 0.5 + epoch * 0.05 * (32 / batch_size))
        experiment.metrics.log(epoch=epoch, train=dict(loss=loss, accuracy=accuracy))
    return accuracy

def hyperparameter_search():
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 32, 64]
    results = []

    for lr, bs in product(learning_rates, batch_sizes):
        with Experiment(
            prefix=f"alice/hyperparameter-search/search-lr{lr}-bs{bs}",
            description=f"Grid search: lr={lr}, batch_size={bs}",
            tags=["grid-search", f"lr-{lr}", f"bs-{bs}"]
        ).run as experiment:
            experiment.params.set(learning_rate=lr, batch_size=bs, optimizer="sgd", epochs=10)
            experiment.log(f"Starting training with lr={lr}, bs={bs}")

            final_accuracy = train_with_config(lr, bs, experiment)
            experiment.log(f"Final accuracy: {final_accuracy:.4f}")
            results.append({"lr": lr, "batch_size": bs, "accuracy": final_accuracy})

    best = max(results, key=lambda x: x["accuracy"])
    print(f"Best: lr={best['lr']}, batch_size={best['batch_size']}, accuracy={best['accuracy']:.4f}")

if __name__ == "__main__":
    hyperparameter_search()
```
