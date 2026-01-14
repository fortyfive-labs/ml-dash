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

## Three Usage Styles

### Context Manager (Recommended)
```python
from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
with Experiment(
    prefix="alice/my-project/my-experiment",
    dash_root=".dash"
).run as experiment:
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)
    # Automatically closed on exit
```

### Decorator Style
```python
from ml_dash import ml_dash_experiment

@ml_dash_experiment(prefix="alice/my-project/my-experiment")
def train_model(experiment):
    experiment.params.set(learning_rate=0.001)
    for epoch in range(10):
        experiment.metrics("train").log(loss=0.5, epoch=epoch)
    return "Training complete!"

result = train_model()
```

### Direct (Manual Control)
```python
from ml_dash import Experiment

experiment = Experiment(
    prefix="alice/my-project/my-experiment",
    dash_root=".dash"
)
experiment.run.start()

try:
    experiment.params.set(learning_rate=0.001)
finally:
    experiment.run.complete()
```

## Local vs Remote Mode

### Local Mode (Filesystem)
```python
with Experiment(
    prefix="alice/my-project/my-experiment",
    dash_root=".dash"  # Storage directory
).run as experiment:
    experiment.log("Using local storage")
```

### Remote Mode (Server)
```python
# First: ml-dash login
with Experiment(
    prefix="alice/my-project/my-experiment",
    dash_url="https://api.dash.ml"  # Token auto-loaded from keychain
).run as experiment:
    experiment.log("Using remote server")
```

## Experiment Metadata

```python
with Experiment(
    prefix="alice/computer-vision/resnet50-imagenet",
    dash_root=".dash",
    description="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    bindrs=["gpu-cluster", "team-a"]
).run as experiment:
    pass
```

## Status Lifecycle

- **RUNNING**: Set when experiment opens
- **COMPLETED**: Set on normal exit
- **FAILED**: Set on exception
- **CANCELLED**: Set manually

### Automatic Status Management
```python
# Normal completion -> COMPLETED
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as experiment:
    experiment.log("Training...")

# Exception -> FAILED
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as experiment:
    raise ValueError("Training failed!")
```

### Manual Status Control
```python
experiment = Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
)
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
with Experiment(
    prefix="alice/ml/long-training",
    dash_root=".dash"
).run as experiment:
    experiment.metrics("train").log(loss=0.5, epoch=1)

# Later - continues same experiment
with Experiment(
    prefix="alice/ml/long-training",
    dash_root=".dash"
).run as experiment:
    experiment.metrics("train").log(loss=0.3, epoch=2)
```
