# Experiments

Experiments are the foundation of ML-Dash. Each experiment represents a single experiment run, containing all your logs, parameters, metrics, and files.

## Three Usage Styles

**Context Manager** (recommended for most cases):

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash") as experiment:
    experiment.log("Training started")
    experiment.parameters().set(learning_rate=0.001)
    # Experiment automatically closed on exit
```

**Decorator** (clean for training functions):

```{code-block} python
:linenos:

from ml_dash import ml_dash_experiment

@ml_dash_experiment(name="my-experiment", project="project")
def train_model(experiment):
    experiment.log("Training started")
    experiment.parameters().set(learning_rate=0.001)

    for epoch in range(10):
        loss = train_epoch()
        experiment.metric("loss").append(value=loss, epoch=epoch)

    return "Training complete!"

result = train_model()
```

**Direct** (manual control):

```{code-block} python
:linenos:

from ml_dash import Experiment

experiment = Experiment(name="my-experiment", project="project",
        local_path=".ml-dash")
experiment.open()

try:
    experiment.log("Training started")
    experiment.parameters().set(learning_rate=0.001)
finally:
    experiment.close()
```

## Local vs Remote Mode

**Local mode** - Zero setup, filesystem storage:

```{code-block} python
:linenos:

with Experiment(
    name="my-experiment",
    project="project",
    local_prefix="./experiments",
        local_path=".ml-dash"
) as experiment:
    experiment.log("Using local storage")
```

**Remote mode** - Team collaboration with server:

```{code-block} python
:linenos:

with Experiment(
    name="my-experiment",
    project="project",
    remote="https://your-server.com",
    user_name="alice"
) as experiment:
    experiment.log("Using remote server")
```

## Experiment Metadata

Add description, tags, and folders for organization:

```{code-block} python
:linenos:

with Experiment(
    name="resnet50-imagenet",
    project="computer-vision",
    local_prefix="./experiments",
    description="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    folder="/experiments/2025/resnet",
        local_path=".ml-dash"
) as experiment:
    experiment.log("Training started")
```

## Resuming Experiments

Experiments use **upsert behavior** - reopen by using the same name:

```{code-block} python
:linenos:

# First run
with Experiment(name="long-training", project="ml",
        local_path=".ml-dash") as experiment:
    experiment.log("Starting epoch 1")
    experiment.metric("loss").append(value=0.5, epoch=1)

# Later - continues same experiment
with Experiment(name="long-training", project="ml",
        local_path=".ml-dash") as experiment:
    experiment.log("Resuming from checkpoint")
    experiment.metric("loss").append(value=0.3, epoch=2)
```

## Available Operations

Once a experiment is open, you can use all ML-Dash features:

```{code-block} python
:linenos:

with Experiment(name="demo", project="test",
        local_path=".ml-dash") as experiment:
    # Logging
    experiment.log("Training started", level="info")

    # Parameters
    experiment.parameters().set(lr=0.001, batch_size=32)

    # Metrics metricing
    experiment.metric("loss").append(value=0.5, epoch=1)

    # File uploads
    experiment.file("model.pth", prefix="/models")
```

## Storage Structure

**Local mode** creates a directory structure:

```
./experiments/
└── project/
    └── my-experiment/
        ├── logs/
        │   └── logs.jsonl
        ├── parameters.json
        ├── metrics/
        │   └── loss/
        │       └── data.jsonl
        └── files/
            └── models/
                └── model.pth
```

**Remote mode** stores data in MongoDB + S3 on your server.

---

**Next:** Learn about [Logging](logging.md) to metric events and progress.
