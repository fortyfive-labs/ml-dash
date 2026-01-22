---
description: Tracking parameters, metrics, logs, and tracks in ML-Dash experiments
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
  - tracks
  - time series
  - loss
  - accuracy
  - trajectory
  - sensor
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

---

## Metrics (Time Series)

### Basic Usage
```python
# Log metrics with keyword arguments
experiment.metrics("train").log(loss=0.5, epoch=1)
experiment.metrics("eval").log(accuracy=0.85, step=100, epoch=1)
```

### Multiple Fields Per Point
```python
experiment.metrics("metrics").log(
    loss=0.5,
    accuracy=0.85,
    learning_rate=0.001,
    epoch=1
)
```

### Multiple Metric Streams
```python
for epoch in range(5):
    experiment.metrics("train").log(loss=0.5 - epoch * 0.1, epoch=epoch)
    experiment.metrics("eval").log(loss=0.6 - epoch * 0.1, epoch=epoch)
```

### Flexible Schema
```python
# Schema can vary between data points
experiment.metrics("flexible").log(field_a=1, field_b=2)
experiment.metrics("flexible").log(field_a=3, field_c=4)
experiment.metrics("flexible").log(field_a=5, field_b=6, field_c=7)
```

### Reading Metrics
```python
result = experiment.metrics("loss").read(start_index=0, limit=10)
for point in result['data']:
    print(f"Index {point['index']}: {point['data']}")
```

### Get Stats
```python
stats = experiment.metrics("train").stats()
print(f"Total points: {stats['totalDataPoints']}")
```

---

## Tracks (Timestamped Multi-Modal Data)

Tracks are for sparse timestamped data like robot trajectories, camera poses, sensor readings.

### Basic Usage
```python
# Timestamp (_ts) is required
experiment.tracks("robot/position").append(
    q=[0.1, 0.2, 0.3],      # joint positions
    e=[0.5, 0.0, 0.6],      # end effector
    _ts=1.0                  # timestamp in seconds
)
```

### Multiple Fields with Same Timestamp
```python
# Entries with same timestamp are merged
experiment.tracks("robot/state").append(q=[0.1, 0.2], _ts=1.0)
experiment.tracks("robot/state").append(v=[0.01, 0.02], _ts=1.0)
# Result: {_ts: 1.0, q: [0.1, 0.2], v: [0.01, 0.02]}
```

### Flushing Tracks
```python
# Flush specific topic
experiment.tracks("robot/position").flush()

# Flush all track topics
experiment.tracks.flush()
```

### Reading Track Data
```python
# Read all data
data = experiment.tracks("robot/position").read()

# Read with time range
data = experiment.tracks("robot/position").read(
    start_timestamp=0.0,
    end_timestamp=10.0
)

# Export formats: json, jsonl, parquet, mocap
parquet_data = experiment.tracks("robot/position").read(format="parquet")
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
- Parameters: `parameters.json`
- Metrics: `metrics/{name}/data.jsonl`
- Logs: `logs/logs.jsonl`
- Tracks: `tracks/{topic}/data.jsonl`

### Remote Mode
- Parameters: MongoDB document
- Metrics: MongoDB (hot) + S3 (cold, after 10K points)
- Logs: MongoDB with timestamp/level indexing
- Tracks: MongoDB with timestamp indexing
