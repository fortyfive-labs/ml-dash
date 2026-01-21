# Experiments

Experiments are the foundation of ML-Dash. Each experiment represents a single experiment run, containing all your logs, parameters, metrics, and files.

## Prefix Format

The prefix is a universal key that identifies your experiment:

```
{owner}/{project}/path.../[name]
```

- **owner**: First segment (e.g., your username)
- **project**: Second segment (e.g., project name)
- **path**: Remaining segments form the folder structure
- **name**: Derived from the last segment

## Three Usage Styles

**Context Manager** (recommended for most cases):

```{code-block} python
:linenos:

from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
with Experiment(prefix="alice/project/my-experiment").run as exp:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001)
    # Experiment automatically closed on exit
```

**Decorator** (clean for training functions):

```{code-block} python
:linenos:

from ml_dash import ml_dash_experiment

@ml_dash_experiment(prefix="alice/project/my-experiment")
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

exp = Experiment(prefix="alice/project/my-experiment")
exp.run.start()

try:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001)
finally:
    exp.run.complete()
```

## Local vs Remote Mode

**Local mode** - Zero setup, filesystem storage:

```{code-block} python
:linenos:

with Experiment(
    prefix="alice/project/my-experiment",
    dash_root=".dash"
).run as exp:
    exp.log("Using local storage")
```

**Remote mode** - Team collaboration with server:

```{code-block} python
:linenos:

with Experiment(
    prefix="alice/project/my-experiment",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Using remote server")
```

## Experiment Metadata

Add description, tags, and bindrs for organization:

```{code-block} python
:linenos:

with Experiment(
    prefix="alice/computer-vision/resnet50-imagenet",
    description="ResNet-50 training with new augmentation",
    tags=["resnet", "imagenet", "baseline"],
    bindrs=["gpu-cluster", "team-a"]
).run as exp:
    exp.log("Training started")
```

**Metadata fields:**
- `description`: Human-readable experiment description
- `tags`: List of tags for categorization (e.g., ["baseline", "production"])
- `bindrs`: List of bindrs for resource/team association (e.g., ["gpu-1", "team-ml"])

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
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Training...")
    # Status automatically set to COMPLETED on exit

# Exception handling - status becomes FAILED
with Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Training...")
    raise ValueError("Training failed!")
    # Status automatically set to FAILED on exception
```

**Manual status control**:

```{code-block} python
:linenos:

from ml_dash import Experiment

exp = Experiment(
    prefix="alice/ml/training",
    dash_url="https://api.dash.ml"
)
exp.run.start()

try:
    exp.log("Training...")
    # ... training code ...
    exp.run.complete()
except KeyboardInterrupt:
    exp.run.cancel()
except Exception as e:
    exp.log(f"Error: {e}")
    exp.run.fail()
```

**Note:** Status updates only work in remote mode. Local mode doesn't track status.

## Resuming Experiments

Experiments use **upsert behavior** - reopen by using the same prefix:

```{code-block} python
:linenos:

# First run
with Experiment(prefix="alice/ml/long-training").run as exp:
    exp.log("Starting epoch 1")
    exp.metrics("train").log(loss=0.5, epoch=1)

# Later - continues same experiment
with Experiment(prefix="alice/ml/long-training").run as exp:
    exp.log("Resuming from checkpoint")
    exp.metrics("train").log(loss=0.3, epoch=2)
```

## Available Operations

Once an experiment is open, you can use all ML-Dash features:

```{code-block} python
:linenos:

with Experiment(prefix="alice/test/demo").run as exp:
    # Logging
    exp.log("Training started", level="info")

    # Parameters
    exp.params.set(lr=0.001, batch_size=32)

    # Metrics tracking
    exp.metrics("train").log(loss=0.5, epoch=1)

    # File uploads
    exp.files("models").save("model.pth")
```

## Storage Structure

**Local mode** creates a directory structure:

```
.dash/
└── alice/                          # owner
    └── project/                    # project
        └── my-experiment/          # experiment name
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

**Next:** Learn about [Logging](logging.md) to track events and progress.
