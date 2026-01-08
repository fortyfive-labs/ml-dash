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
from ml_dash.auto_start import dxp

with dxp.run:
    dxp.log("Training started", level="info")
    dxp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        dxp.metrics("train").log(loss=loss, epoch=epoch)

    dxp.log("Training completed", level="info")
```

## Quick Start - Local Mode (No Auth)

```python
from ml_dash import Experiment

exp = Experiment(prefix="alice/tutorial/my-experiment")

with exp.run:
    exp.log("Training started", level="info")
    exp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = 1.0 - epoch * 0.08
        exp.metrics("train").log(loss=loss, epoch=epoch)
```

Data stored in `.dash/alice/tutorial/my-experiment/`.

## Pre-configured Singleton (dxp)

```python
from ml_dash.auto_start import dxp

# Ready to use immediately
dxp.params.set(learning_rate=0.001)
dxp.metrics("train").log(loss=0.5, accuracy=0.9)
dxp.files.save_torch(model, "model.pt")
```

## Data Storage Structure

```
.dash/
└── owner/
    └── project/
        └── experiment/
            ├── logs/logs.jsonl
            ├── parameters/parameters.json
            └── metrics/train.jsonl
```
