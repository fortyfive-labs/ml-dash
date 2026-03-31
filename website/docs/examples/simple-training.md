---
sidebar_label: Simple Training Loop
---

# Simple Training Loop

A basic training loop with logging, parameters, and metrics tracking.

```python
"""Simple training loop example."""
import random
from ml_dash import Experiment

def train_simple_model():
    with Experiment(
        prefix="alice/tutorials/simple-training",
        description="Simple training example",
        tags=["tutorial", "simple"]
    ).run as experiment:
        experiment.params.set(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            model="simple_nn"
        )

        experiment.log("Starting training", level="info")

        for epoch in range(10):
            train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
            val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
            accuracy = min(0.95, 0.5 + epoch * 0.05)

            experiment.metrics.log(
                epoch=epoch,
                train=dict(loss=train_loss, accuracy=accuracy),
                eval=dict(loss=val_loss, accuracy=accuracy)
            )

            experiment.log(
                f"Epoch {epoch + 1}/10 complete",
                level="info",
                metadata={"train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy}
            )

        experiment.log("Training complete!", level="info")

if __name__ == "__main__":
    train_simple_model()
```
