# Experiments

Experiments are the foundation of ML-Dash. Each experiment represents a single experiment run, containing all your logs, parameters, metrics, and files.

## Three Usage Styles

**Context Manager** (recommended for most cases):

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="my-experiment", project="project",
        ).run as experiment:
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)
    # Experiment automatically closed on exit
```

**Decorator** (clean for training functions):

```{code-block} python
:linenos:

from ml_dash import ml_dash_experiment

@ml_dash_experiment(prefix="my-experiment", project="project")
def train_model(experiment):
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)

    for epoch in range(10):
        loss = train_epoch()
        experiment.metrics("train").log(loss=loss, epoch=epoch)

    return "Training complete!"

result = train_model()
```

**Direct** (manual control):

```{code-block} python
:linenos:

from ml_dash import Experiment

experiment = Experiment(prefix="my-experiment", project="project",
        )
experiment.run.start()

try:
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)
finally:
    experiment.run.complete()
```

## Local vs Remote Mode

**Local mode** - Zero setup, filesystem storage:

```{code-block} python
:linenos:

with Experiment(
    prefix="my-experiment",
    project="project",
    
).run as experiment:
    experiment.log("Using local storage")
```

**Remote mode** - Team collaboration with server:

```{code-block} python
:linenos:

with Experiment(
    prefix="my-experiment",
    project="project",
    remote="https://api.dash.ml",
    user_name="alice"
).run as experiment:
    experiment.log("Using remote server")
```

## Experiment Metadata

Add description, tags, bindrs, and folders for organization:

```{code-block} python
:linenos:

with Experiment(
    prefix="resnet50-imagenet",
    project="computer-vision",
    
    description="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    bindrs=["gpu-cluster", "team-a"],
    folder="/experiments/2025/resnet"
).run as experiment:
    experiment.log("Training started")
```

**Metadata fields:**
- `description`: Human-readable experiment description
- `tags`: List of tags for categorization (e.g., ["baseline", "production"])
- `bindrs`: List of bindrs for resource/team association (e.g., ["gpu-1", "team-ml"])
- `prefix`: Hierarchical folder path for organization (referred to as "folder" in the directory structure)

## Experiment Status Lifecycle

Experiments automatically track their status through the lifecycle:

- **RUNNING**: Automatically set when experiment opens
- **COMPLETED**: Set when experiment closes normally
- **FAILED**: Set when exception occurs during experiment
- **CANCELLED**: Can be set manually

**Automatic status management** (recommended):

```{code-block} python
:linenos:

# Normal completion - status becomes COMPLETED
with Experiment(prefix="training", project="ml",
        remote="https://api.dash.ml").run as experiment:
    experiment.log("Training...")
    # Status automatically set to COMPLETED on exit

# Exception handling - status becomes FAILED
with Experiment(prefix="training", project="ml",
        remote="https://api.dash.ml").run as experiment:
    experiment.log("Training...")
    raise ValueError("Training failed!")
    # Status automatically set to FAILED on exception
```

**Manual status control**:

```{code-block} python
:linenos:

from ml_dash import Experiment

experiment = Experiment(prefix="training", project="ml",
        remote="https://api.dash.ml")
experiment.run.start()

try:
    experiment.log("Training...")
    # ... training code ...
    experiment.run.complete()
except KeyboardInterrupt:
    experiment.run.cancel()
except Exception as e:
    experiment.log(f"Error: {e}")
    experiment.run.fail()
```

**Note:** Status updates only work in remote mode. Local mode doesn't track status.

## Resuming Experiments

Experiments use **upsert behavior** - reopen by using the same name:

```{code-block} python
:linenos:

# First run
with Experiment(prefix="long-training", project="ml",
        ).run as experiment:
    experiment.log("Starting epoch 1")
    experiment.metrics("train").log(loss=0.5, epoch=1)

# Later - continues same experiment
with Experiment(prefix="long-training", project="ml",
        ).run as experiment:
    experiment.log("Resuming from checkpoint")
    experiment.metrics("train").log(loss=0.3, epoch=2)
```

## Available Operations

Once a experiment is open, you can use all ML-Dash features:

```{code-block} python
:linenos:

with Experiment(prefix="demo", project="test",
        ).run as experiment:
    # Logging
    experiment.log("Training started", level="info")

    # Parameters
    experiment.params.set(lr=0.001, batch_size=32)

    # Metrics tracking
    experiment.metrics("train").log(loss=0.5, epoch=1)

    # File uploads
    experiment.files("models").save("model.pth")
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
        │   └── train/
        │       └── data.jsonl
        └── files/
            └── models/
                └── 7218065541365719/
                    └── model.pth
```

Files are stored under `files/{prefix}/{snowflake_id}/{filename}` where:
- `prefix`: Logical path (e.g., "models", "configs")
- `snowflake_id`: Unique identifier for the file
- `filename`: Original filename

**Remote mode** stores data in MongoDB + S3 on your server.

---

**Next:** Learn about [Logging](logging.md) to metric events and progress.
