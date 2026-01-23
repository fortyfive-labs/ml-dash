---
description: Experiment class configuration, usage patterns, local vs remote modes, and lifecycle management
globs:
  - "**/*.py"
  - "**/train*.py"
  - "**/experiment*.py"
keywords:
  - Experiment
  - context manager
  - decorator
  - local mode
  - remote mode
  - experiment lifecycle
  - run.start
  - run.complete
  - run.fail
  - status
  - RUNNING
  - COMPLETED
  - FAILED
---

# ML-Dash Experiment Configuration

## Two Usage Styles

### Context Manager (Recommended)
```python
from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
with Experiment(prefix="alice/my-project/my-experiment").run as experiment:
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)
    experiment.metrics("train").log(loss=0.5, epoch=0)
    # Automatically closed on exit
```

### Direct (Manual Control)
```python
from ml_dash import Experiment

experiment = Experiment(prefix="alice/my-project/my-experiment")
experiment.run.start()

try:
    experiment.params.set(learning_rate=0.001)
    experiment.metrics("train").log(loss=0.5, epoch=0)
finally:
    experiment.run.complete()
```

## Local vs Remote Mode

### Local Mode (Filesystem)
```python
from ml_dash import Experiment

with Experiment(
    prefix="alice/my-project/my-experiment",
    dash_root=".dash"  # Storage directory
).run as experiment:
    experiment.log("Using local storage")
```

### Remote Mode (Server)
```bash
# First authenticate
ml-dash login
```

```python
from ml_dash import Experiment

# Token auto-loaded from keychain after login
with Experiment(
    prefix="alice/my-project/my-experiment",
    dash_url="https://api.dash.ml"
).run as experiment:
    experiment.log("Using remote server")
```

## Experiment Metadata

```python
with Experiment(
    prefix="alice/computer-vision/resnet50-imagenet",
    readme="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
).run as experiment:
    experiment.log("Training with metadata")
```

## Status Lifecycle

- **RUNNING**: Set when experiment opens
- **COMPLETED**: Set on normal exit
- **FAILED**: Set on exception

### Automatic Status Management
```python
# Normal completion -> COMPLETED
with Experiment(prefix="alice/ml/training").run as experiment:
    experiment.log("Training...")

# Exception -> FAILED
with Experiment(prefix="alice/ml/training").run as experiment:
    raise ValueError("Training failed!")  # Status set to FAILED
```

### Manual Status Control
```python
experiment = Experiment(prefix="alice/ml/training")
experiment.run.start()

try:
    # training...
    experiment.run.complete()
except KeyboardInterrupt:
    experiment.run.cancel()
except Exception as e:
    experiment.run.fail()
```

## Resuming Experiments

Experiments use upsert behavior - reopen by using the same prefix:

```python
# First run
with Experiment(prefix="alice/ml/long-training").run as experiment:
    experiment.metrics("train").log(loss=0.5, epoch=1)

# Later - continues same experiment
with Experiment(prefix="alice/ml/long-training").run as experiment:
    experiment.metrics("train").log(loss=0.3, epoch=2)
```

## Using RUN Configuration

```python
from ml_dash.run import RUN
from ml_dash.auto_start import dxp

# Configure prefix before use
RUN.prefix = "geyang/scratch/some-experiment"

# dxp will use RUN configuration
with dxp.run:
    dxp.metrics("train").log(step=0, loss=1.0)
```

## Background Buffering

ML-Dash automatically buffers writes in the background for performance:

```python
with Experiment(prefix="alice/ml/fast-training").run as experiment:
    for i in range(1000):
        # Non-blocking writes buffered in background
        experiment.metrics("train").log(loss=1.0 / (i + 1), step=i)

    # Manual flush if needed
    experiment.flush()  # Force immediate write
# Automatic flush on context exit
```

Buffer configuration via environment variables:
- `ML_DASH_FLUSH_INTERVAL`: Time-based flush interval (default: 5.0 seconds)
- `ML_DASH_METRIC_BATCH_SIZE`: Max metric points per batch (default: 100)
- `ML_DASH_LOG_BATCH_SIZE`: Max logs per batch (default: 100)
