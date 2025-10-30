# Basic Training Loop

A complete example showing how to metric a simple training loop with DreamLake.

## The Scenario

You're training a neural network and want to metric:
- Hyperparameters (learning rate, batch size, epochs)
- Training progress with logs
- Metrics over time (loss, accuracy)
- Final model file

## Complete Code

```{code-block} python
:linenos:

import random
from ml_dash import Experiment

def train_simple_model():
    """Train a simple model and metric everything with DreamLake."""

    with Experiment(
        name="simple-training",
        project="tutorials",
        description="Basic training loop example",
        tags=["tutorial", "simple"],
        local_path=".ml-dash"
    ) as experiment:
        # Metric hyperparameters
        experiment.parameters().set(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            optimizer="adam",
            model="simple_nn"
        )

        experiment.log("Starting training", level="info")

        # Training loop
        for epoch in range(10):
            # Simulate training
            train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
            val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
            accuracy = min(0.95, 0.5 + epoch * 0.05)

            # Metric metrics
            experiment.metric("train_loss").append(value=train_loss, epoch=epoch)
            experiment.metric("val_loss").append(value=val_loss, epoch=epoch)
            experiment.metric("accuracy").append(value=accuracy, epoch=epoch)

            # Log progress
            experiment.log(
                f"Epoch {epoch + 1}/10 complete",
                level="info",
                metadata={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy
                }
            )

        # Save model file
        # In real code: torch.save(model.state_dict(), "model.pth")
        with open("model.pth", "w") as f:
            f.write("model weights")

        experiment.file(
            file_path="model.pth",
            prefix="/models",
            description="Final trained model",
            tags=["final"]
        ).save()

        experiment.log("Training complete!", level="info")
        print(f"✓ Experiment metriced successfully")

if __name__ == "__main__":
    train_simple_model()
```

## What Gets Metriced

**Parameters:** All hyperparameters are saved once
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 10,
  "optimizer": "adam",
  "model": "simple_nn"
}
```

**Logs:** Progress messages with timestamps
```json
{"timestamp": "2025-10-29T10:30:00Z", "level": "info", "message": "Starting training"}
{"timestamp": "2025-10-29T10:30:05Z", "level": "info", "message": "Epoch 1/10 complete", "metadata": {"train_loss": 0.95, "val_loss": 1.15, "accuracy": 0.5}}
```

**Metrics:** Metrics over time
```json
{"index": 0, "data": {"value": 0.95, "epoch": 0}}
{"index": 1, "data": {"value": 0.52, "epoch": 1}}
{"index": 2, "data": {"value": 0.35, "epoch": 2}}
```

**Files:** Model saved with metadata
- Filename: `model.pth`
- Path: `/models`
- Checksum: SHA256 hash
- Tags: `["final"]`

## Key Patterns

**Use experiment context manager** - Automatic cleanup:
```python
with Experiment(name="...", project="...") as experiment:
    # Your code here
```

**Metric parameters once** - At the start:
```python
experiment.parameters().set(learning_rate=0.001, batch_size=32)
```

**Metric metrics in the loop** - Every epoch:
```python
experiment.metric("loss").append(value=loss, epoch=epoch)
```

**Log important events** - Progress and completion:
```python
experiment.log("Epoch 1/10 complete", metadata={"loss": 0.5})
```

---

**Next:** See [Hyperparameter Search](hyperparameter-search.md) for metricing multiple experiments.
