---
description: ML-Dash installation, quick start, and basic usage patterns
globs:
  - "**/*.py"
  - "**/requirements*.txt"
  - "**/pyproject.toml"
  - "**/setup.py"
keywords:
  - install
  - setup
  - getting started
  - quick start
  - pip install
  - uv add
  - ml-dash
  - dxp
  - import ml_dash
---

# ML-Dash Getting Started

## Installation

```bash
pip install ml-dash
# or
uv add ml-dash
```

## Quick Start - Remote Mode (Recommended)

### 1. Authenticate
```bash
ml-dash login
```
Opens browser for secure OAuth2 authentication. Token stored in system keychain.

### 2. Start Tracking

```python
from ml_dash import dxp

with dxp.run:
    dxp.log().info("Training started")
    dxp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        dxp.metrics("loss").append(value=loss, epoch=epoch)

    dxp.log().info("Training completed")
```

## Quick Start - Local Mode (No Auth)

```python
from ml_dash import Experiment

with Experiment(name="my-experiment", project="tutorial",
        local_path=".dash").run as experiment:
    experiment.log().info("Training started")
    experiment.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        experiment.metrics("loss").append(value=loss, epoch=epoch)
```

Data stored in `.dash/tutorial/my-experiment/`.

## Pre-configured Singleton (dxp)

```python
from ml_dash import dxp  # or from ml_dash.auto_start import dxp

# Ready to use immediately
dxp.params.set(learning_rate=0.001)
dxp.metrics.append(loss=0.5, accuracy=0.9)
dxp.files.save_torch(model, "model.pt")
```

## Data Storage Structure

```
.dash/
└── project/
    └── experiment/
        ├── logs/logs.jsonl
        ├── parameters/parameters.json
        └── metrics/loss.jsonl
```
