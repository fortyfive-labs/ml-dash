---
name: getting-started
description: ML-Dash installation, quick start, and basic usage patterns (plugin:ml-dash@ml-dash)
---

# ML-Dash Getting Started

## Installation

### Python Package

```bash
pip install ml-dash
# or
uv add ml-dash
```

### Claude Plugin (Skills)

```bash
# Add ml-dash marketplace
/plugin marketplace add fortyfive-labs/ml-dash

# Install ml-dash skills
/plugin install ml-dash@ml-dash
```

Then enable in Settings > Plugins > `ml-dash@ml-dash`

## Quick Start - Local Mode

```python
from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
exp = Experiment(prefix="alice/tutorial/my-experiment")

with exp.run:
    exp.log("Training started", level="info")
    exp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        exp.metrics("train").log(loss=loss, epoch=epoch)

    exp.log("Training completed", level="info")
```

Data stored in `.dash/alice/tutorial/my-experiment/`.

## Quick Start - Remote Mode

```bash
# First authenticate
ml-dash login
```

```python
from ml_dash.auto_start import dxp

with dxp.run:
    dxp.log("Training started", level="info")
    dxp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        dxp.metrics("train").log(loss=loss, epoch=epoch)

    dxp.log("Training completed", level="info")
```

## Pre-configured Singleton (dxp)

```python
from ml_dash.auto_start import dxp
from ml_dash.run import RUN

# Configure prefix before use
RUN.prefix = "geyang/scratch/my-experiment"

with dxp.run:
    dxp.params.set(learning_rate=0.001)
    dxp.metrics("train").log(loss=0.5, accuracy=0.9)
    dxp.files("models").save_torch(model, to="model.pt")
```

## Key APIs

### Metrics (Time Series)
```python
# Log metrics with keyword arguments
experiment.metrics("train").log(loss=0.5, epoch=1)
experiment.metrics("eval").log(accuracy=0.85, epoch=1)
```

### Parameters
```python
experiment.params.set(
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam"
)
```

### Logging
```python
experiment.log("Training started", level="info")
experiment.log("GPU memory low", level="warn")
experiment.log("Failed to load", level="error")
```

### Files
```python
# Save various file types
experiment.files("models").save("model.pth")
experiment.files("configs").save({"lr": 0.001}, to="config.json")
experiment.files("data").save_text("content", to="notes.txt")
```

### Tracks (Timestamped Data)
```python
# For robot trajectories, sensor data, etc.
experiment.tracks("robot/position").append(
    q=[0.1, 0.2, 0.3],
    _ts=1.0  # timestamp required
)
```

## Data Storage Structure

```
.dash/
└── owner/
    └── project/
        └── experiment/
            ├── experiment.json
            ├── parameters.json
            ├── logs/logs.jsonl
            ├── metrics/train/data.jsonl
            └── files/
```
