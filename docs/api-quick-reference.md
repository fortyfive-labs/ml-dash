# ML-Dash - API Quick Reference

Quick reference for common ML-Dash operations.

## Experiment Creation

```python
from ml_dash import Experiment

# Local mode
with Experiment(
    name="experiment-name",
    project="project-name",
    local_prefix="./data",
        local_path=".ml-dash"
) as experiment:
    # Your code here
    pass

# Remote mode (with username - auto-generates API key)
with Experiment(
    name="experiment-name",
    project="project-name",
    remote="https://api.dash.ml",
    user_name="your-username"
) as experiment:
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
experiment.parameters().set(
    learning_rate=0.001,
    batch_size=32
)

# Set parameters (dictionary - supports nested)
experiment.parameters().set(**{
    "model": {
        "architecture": "resnet50",
        "layers": 50
    }
})
# Stored as: {"model.architecture": "resnet50", "model.layers": 50}

# Update parameters
experiment.parameters().set(learning_rate=0.0001)
```

## Metrics (Time-Series Metrics)

```python
# Append single data point
experiment.metric("train_loss").append(value=0.5, epoch=1)

# Flexible schema
experiment.metric("metrics").append(
    loss=0.5,
    accuracy=0.85,
    epoch=1
)

# Batch append
experiment.metric("loss").append_batch([
    {"value": 0.5, "epoch": 1},
    {"value": 0.4, "epoch": 2},
    {"value": 0.3, "epoch": 3}
])

# Read data
result = experiment.metric("loss").read(start_index=0, limit=10)
for point in result['data']:
    print(f"Index {point['index']}: {point['data']}")

# Get statistics
stats = experiment.metric("loss").stats()
print(f"Total points: {stats['totalDataPoints']}")

# List all metrics
metrics = experiment.metric("loss").list_all()
for metric in metrics:
    print(f"{metric['name']}: {metric['totalDataPoints']} points")
```

## Files

```python
# Upload file
experiment.file(
    file_prefix="model.pth",
    prefix="models/",
    description="Trained model",
    tags=["final", "best"]
).save()

# Upload with metadata
experiment.file(
    file_prefix="model.pth",
    prefix="models/checkpoints/",
    metadata={"epoch": 50, "accuracy": 0.95}
).save()

# List files
files = experiment.file().list()
for file in files:
    print(f"{file['prefix']}{file['filename']}")
```

## Complete Example

```python
from ml_dash import Experiment

with Experiment(
    name="mnist-training",
    project="computer-vision",
    local_prefix="./experiments",
        local_path=".ml-dash"
) as experiment:
    # Configuration
    experiment.parameters().set(
        learning_rate=0.001,
        batch_size=64,
        epochs=10
    )

    experiment.log("Training started", level="info")

    # Training loop
    for epoch in range(10):
        # Train
        train_loss, val_loss, accuracy = train_one_epoch()

        # Metric metrics
        experiment.metric("train_loss").append(value=train_loss, epoch=epoch)
        experiment.metric("val_loss").append(value=val_loss, epoch=epoch)
        experiment.metric("accuracy").append(value=accuracy, epoch=epoch)

        # Log progress
        experiment.log(
            f"Epoch {epoch + 1}/10 complete",
            metadata={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy
            }
        )

    # Save model
    save_model("model.pth")
    experiment.file(file_prefix="model.pth", prefix="models/").save()

    experiment.log("Training complete!", level="info")
```

## Common Patterns

### Training with Checkpoints

```python
with Experiment(...) as experiment:
    best_acc = 0
    for epoch in range(epochs):
        train()
        acc = validate()

        experiment.metric("accuracy").append(value=acc, epoch=epoch)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(f"checkpoint_{epoch}.pth")
            experiment.file(
                file_path=f"checkpoint_{epoch}.pth",
                prefix="checkpoints/",
                tags=["best"]
            ).save()
```

### Hyperparameter Search

```python
for lr in [0.1, 0.01, 0.001]:
    for bs in [32, 64, 128]:
        with Experiment(name=f"search-lr{lr}-bs{bs}", ...) as experiment:
            experiment.parameters().set(
                learning_rate=lr,
                batch_size=bs
            )

            accuracy = train(lr, bs)
            experiment.metric("accuracy").append(value=accuracy)
```

### Progress Logging

```python
with Experiment(...) as experiment:
    total = 1000
    for i in range(total):
        process_item(i)

        if i % 100 == 0:
            percent = (i / total) * 100
            experiment.log(
                f"Progress: {percent}%",
                metadata={"processed": i, "total": total}
            )
```

## Data Storage

### Local Mode

```
.ml-dash/
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
cat .ml-dash/project/experiment/logs.jsonl

# View parameters
cat .ml-dash/project/experiment/parameters.json

# View metric data
cat .ml-dash/project/experiment/metrics/train_loss/data.jsonl

# List files
ls .ml-dash/project/experiment/files/
```

## See Also

- [Complete Examples](complete-examples.md)
- [Runnable Examples](examples.md)
