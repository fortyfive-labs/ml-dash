---
sidebar_label: Logging & Debugging
---

# Logging & Debugging

Comprehensive logging for debugging training runs.

```python
"""Debugging example with comprehensive logging."""
from ml_dash import Experiment
import random

def train_with_debug():
    with Experiment(
        prefix="alice/debugging/debug-training",
        description="Training with debug logging",
        tags=["debug"]
    ).run as experiment:
        experiment.params.set(learning_rate=0.001, batch_size=32, model="debug_net")
        experiment.log("Training experiment started", level="info")
        experiment.log("Initializing model", level="debug")

        for epoch in range(5):
            experiment.log(f"Starting epoch {epoch + 1}", level="debug")
            loss = 1.0 / (epoch + 1)

            if epoch == 2:
                experiment.log(
                    "Learning rate may be too high",
                    level="warn",
                    metadata={"current_lr": 0.001, "suggested_lr": 0.0001}
                )

            if random.random() < 0.2:
                experiment.log(
                    "Gradient clipping applied",
                    level="warn",
                    metadata={"gradient_norm": 15.5, "max_norm": 10.0}
                )

            experiment.metrics("train").log(loss=loss, epoch=epoch)
            experiment.log(f"Epoch {epoch + 1} complete", level="info", metadata={"loss": loss})

        experiment.log("Training complete", level="info")

if __name__ == "__main__":
    train_with_debug()
```
