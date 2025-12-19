---
description: Tracking parameters, metrics, and logs in ML-Dash experiments
globs:
  - "**/*.py"
  - "**/train*.py"
  - "**/config*.py"
keywords:
  - params
  - parameters
  - hyperparameters
  - metrics
  - log
  - logging
  - append
  - append_batch
  - set
  - time series
  - loss
  - accuracy
---

# ML-Dash Data Tracking

## Parameters (Hyperparameters)

### Basic Usage
```python
experiment.params.set(
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam",
    epochs=100
)
```

### Nested Parameters (Auto-flattened)
```python
experiment.params.set(**{
    "model": {
        "architecture": "resnet50",
        "pretrained": True
    },
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    }
})
# Stored as: model.architecture, model.pretrained, optimizer.type, etc.
```

### From Config Files
```python
import json
with open("config.json") as f:
    config = json.load(f)
experiment.params.set(**config)
```

### From argparse
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()
experiment.params.set(**vars(args))
```

### From Dataclass
```python
from dataclasses import dataclass, asdict

@dataclass
class Config:
    learning_rate: float = 0.001
    batch_size: int = 32

experiment.params.set(**asdict(Config()))
```

---

## Metrics (Time Series)

### Basic Usage
```python
experiment.metrics("train_loss").append(value=0.5, epoch=1)
experiment.metrics("accuracy").append(value=0.85, step=100, epoch=1)
```

### Flexible Schema
```python
# Multiple fields per point
experiment.metrics("metrics").append(
    loss=0.5,
    accuracy=0.85,
    learning_rate=0.001,
    epoch=1
)

# With timestamp
import time
experiment.metrics("system").append(
    cpu_percent=45.2,
    memory_mb=1024,
    timestamp=time.time()
)
```

### Batch Append (Better Performance)
```python
experiment.metrics("train_loss").append_batch([
    {"value": 0.5, "step": 1, "epoch": 1},
    {"value": 0.45, "step": 2, "epoch": 1},
    {"value": 0.40, "step": 3, "epoch": 1},
])
```

### Batch Collection Pattern
```python
batch = []
for step in range(1000):
    loss = train_step()
    batch.append({"value": loss, "step": step})

    if len(batch) >= 100:
        experiment.metrics("train_loss").append_batch(batch)
        batch = []

if batch:  # Remaining
    experiment.metrics("train_loss").append_batch(batch)
```

### Reading Metrics
```python
result = experiment.metrics("loss").read(start_index=0, limit=10)
for point in result['data']:
    print(f"Index {point['index']}: {point['data']}")
```

---

## Logging

### Basic Usage
```python
experiment.log("Training started")
experiment.log("Model: ResNet-50", level="info")
experiment.log("GPU memory low", level="warn")
experiment.log("Failed to load", level="error")
```

### Log Levels
`debug`, `info` (default), `warn`, `error`, `fatal`

### Structured Metadata
```python
experiment.log(
    "Epoch completed",
    level="info",
    metadata={
        "epoch": 5,
        "train_loss": 0.234,
        "val_loss": 0.456
    }
)
```

### Training Loop Pattern
```python
for epoch in range(10):
    train_loss = train_one_epoch()
    val_loss = validate()

    experiment.log(
        f"Epoch {epoch + 1} complete",
        level="info",
        metadata={"train_loss": train_loss, "val_loss": val_loss}
    )
```

### Error Tracking
```python
try:
    result = risky_operation()
except Exception as e:
    experiment.log(
        f"Operation failed: {e}",
        level="error",
        metadata={"error_type": type(e).__name__}
    )
    raise
```

---

## Storage Format

### Local Mode
- Parameters: `parameters/parameters.json`
- Metrics: `metrics/{name}/data.jsonl`
- Logs: `logs/logs.jsonl`

### Remote Mode
- Parameters: MongoDB document
- Metrics: MongoDB (hot) + S3 (cold, after 10K points)
- Logs: MongoDB with timestamp/level indexing
