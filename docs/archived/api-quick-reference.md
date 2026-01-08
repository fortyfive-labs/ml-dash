# ML-Dash - API Quick Reference

Quick reference for common ML-Dash operations.

## Experiment Creation

```python
from ml_dash import Experiment

# Local mode
exp = Experiment(
    prefix="owner/project/experiment-name",
    local_path=".dash"
)

with exp.run:
    # Your code here
    pass

# Remote mode (with username - auto-generates API key)
exp = Experiment(
    prefix="owner/project/experiment-name",
    remote="https://api.dash.ml",
    user_name="your-username"
)

with exp.run:
    # Your code here
    pass
```

**Note**: When using `user_name`, an API key is automatically generated from the username. This is useful for development when a full authentication service isn't available yet.

## Logging

```python
# Simple log
experiment.log("Training started")

# Log with level
experiment.log("Error occurred", level="error")

# Log with metadata
experiment.log(
    "Epoch complete",
    level="info",
    metadata={"epoch": 5, "loss": 0.234}
)
```

**Levels**: `debug`, `info`, `warn`, `error`, `fatal`

## Parameters

```python
# Set parameters (keyword arguments)
exp.params.set(
    learning_rate=0.001,
    batch_size=32
)

# Set parameters (dictionary - supports nested)
exp.params.set(**{
    "model": {
        "architecture": "resnet50",
        "layers": 50
    }
})
# Stored as: {"model.architecture": "resnet50", "model.layers": 50}

# Update parameters
exp.params.set(learning_rate=0.0001)
```

## Metrics (Time-Series Metrics)

```python
# Log single data point
experiment.metrics("train").log(loss=0.5, epoch=1)

# Flexible schema
experiment.metrics("train").log(
    loss=0.5,
    accuracy=0.85,
    epoch=1
)

# Batch log
experiment.metrics("train").log_batch([
    {"loss": 0.5, "epoch": 1},
    {"loss": 0.4, "epoch": 2},
    {"loss": 0.3, "epoch": 3}
])

# Read data
result = experiment.metrics("train").read(start_index=0, limit=10)
for point in result['data']:
    print(f"Index {point['index']}: {point['data']}")

# Get statistics
stats = experiment.metrics("train").stats()
print(f"Total points: {stats['totalDataPoints']}")

# List all metrics
metrics = experiment.metrics("train").list_all()
for metric in metrics:
    print(f"{metric['name']}: {metric['totalDataPoints']} points")
```

## Files

```python
# Upload file
experiment.files(
    file_prefix="model.pth",
    prefix="models/",
    description="Trained model",
    tags=["final", "best"]
).save()

# Upload with metadata
experiment.files(
    file_prefix="model.pth",
    prefix="models/checkpoints/",
    metadata={"epoch": 50, "accuracy": 0.95}
).save()

# List files
files = experiment.files().list()
for file in files:
    print(f"{file['prefix']}{file['filename']}")
```

## Complete Example

```python
from ml_dash import Experiment

exp = Experiment(
    prefix="owner/computer-vision/mnist-training",
    local_path=".dash/owner/computer-vision/mnist-training"
)

with exp.run:
    # Configuration
    exp.params.set(
        learning_rate=0.001,
        batch_size=64,
        epochs=10
    )

    exp.log("Training started", level="info")

    # Training loop
    for epoch in range(10):
        # Train
        train_loss, val_loss, accuracy = train_one_epoch()

        # Log metrics
        exp.metrics("train").log(loss=train_loss, epoch=epoch)
        exp.metrics("val").log(loss=val_loss, epoch=epoch)
        exp.metrics("eval").log(accuracy=accuracy, epoch=epoch)

        # Log progress
        exp.log(
            f"Epoch {epoch + 1}/10 complete",
            metadata={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy
            }
        )

    # Save model
    save_model("model.pth")
    exp.files(file_prefix="model.pth", prefix="models/").save()

    exp.log("Training complete!", level="info")
```

## Common Patterns

### Training with Checkpoints

```python
exp = Experiment(...)

with exp.run:
    best_acc = 0
    for epoch in range(epochs):
        train()
        acc = validate()

        exp.metrics("eval").log(accuracy=acc, epoch=epoch)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(f"checkpoint_{epoch}.pth")
            exp.files(
                file_path=f"checkpoint_{epoch}.pth",
                prefix="checkpoints/",
                tags=["best"]
            ).save()
```

### Hyperparameter Search

```python
for lr in [0.1, 0.01, 0.001]:
    for bs in [32, 64, 128]:
        exp = Experiment(prefix=f"owner/project/search-lr{lr}-bs{bs}", ...)

        with exp.run:
            exp.params.set(
                learning_rate=lr,
                batch_size=bs
            )

            accuracy = train(lr, bs)
            exp.metrics("eval").log(accuracy=accuracy)
```

### Progress Logging

```python
exp = Experiment(...)

with exp.run:
    total = 1000
    for i in range(total):
        process_item(i)

        if i % 100 == 0:
            percent = (i / total) * 100
            exp.log(
                f"Progress: {percent}%",
                metadata={"processed": i, "total": total}
            )
```

## Data Storage

### Local Mode

```
.dash/
└── project-name/
    └── experiment-name/
        ├── logs.jsonl              # Log entries
        ├── parameters.json         # Parameters
        ├── metrics/                 # Time-series data
        │   └── train_loss/
        │       ├── data.jsonl
        │       └── metadata.json
        └── files/                  # Uploaded files
            └── models/
                └── model.pth
```

### Remote Mode

- **MongoDB**: Logs, parameters, metric metadata, file metadata
- **S3**: Uploaded files, archived logs, metric chunks

## Useful Commands

```bash
# View logs
cat .dash/owner/project/experiment/logs.jsonl

# View parameters
cat .dash/owner/project/experiment/parameters.json

# View metric data
cat .dash/owner/project/experiment/metrics/train_loss/data.jsonl

# List files
ls .dash/owner/project/experiment/files/
```

## See Also

- [Complete Examples](complete-examples.md)
- [Runnable Examples](examples.md)
