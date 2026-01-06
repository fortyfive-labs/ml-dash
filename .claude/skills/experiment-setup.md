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

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)
    # Automatically closed on exit
```

### Decorator Style
```python
from ml_dash import ml_dash_experiment

@ml_dash_experiment(name="my-experiment", project="project")
def train_model(experiment):
    experiment.params.set(learning_rate=0.001)
    for epoch in range(10):
        experiment.metrics("loss").append(value=0.5, epoch=epoch)
    return "Training complete!"

result = train_model()
```

### Direct (Manual Control)
```python
from ml_dash import Experiment

experiment = Experiment(name="my-experiment", project="project",
        local_path=".dash")
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
    name="my-experiment",
    project="project",
    local_path=".dash"  # Storage directory
).run as experiment:
    experiment.log("Using local storage")
```

### Remote Mode (Server)
```python
# First: ml-dash login
with Experiment(
    name="my-experiment",
    project="project",
    remote="https://api.dash.ml",
    user_name="alice"
).run as experiment:
    experiment.log("Using remote server")
```

## Experiment Metadata

```python
with Experiment(
    name="resnet50-imagenet",
    project="computer-vision",
    local_path=".dash",
    description="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    bindrs=["gpu-cluster", "team-a"],
    folder="/experiments/2025/resnet"
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
with Experiment(name="training", project="ml",
        remote="https://api.dash.ml").run as experiment:
    experiment.log("Training...")

# Exception -> FAILED
with Experiment(name="training", project="ml",
        remote="https://api.dash.ml").run as experiment:
    raise ValueError("Training failed!")
```

### Manual Status Control
```python
experiment = Experiment(name="training", project="ml",
        remote="https://api.dash.ml")
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

Experiments use upsert behavior - reopen by using the same name:

```python
# First run
with Experiment(name="long-training", project="ml",
        local_path=".dash").run as experiment:
    experiment.metrics("loss").append(value=0.5, epoch=1)

# Later - continues same experiment
with Experiment(name="long-training", project="ml",
        local_path=".dash").run as experiment:
    experiment.metrics("loss").append(value=0.3, epoch=2)
```
