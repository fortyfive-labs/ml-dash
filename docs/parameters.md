# Parameters

Metric hyperparameters, configuration values, and experiment settings. Parameters are static key-value pairs that define your experiment.

## Basic Usage

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        optimizer="adam",
        epochs=100
    )
```

## Nested Parameters

Use nested dictionaries - they're automatically flattened with dot notation:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(**{
        "model": {
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 1000
        },
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 0.0001
        }
    })

    # Stored as:
    # model.architecture = "resnet50"
    # model.pretrained = True
    # optimizer.type = "adam"
    # ...
```

## Class Objects Support

Pass configuration classes directly (perfect for `params-proto`):

```{code-block} python
:linenos:

class TrainingConfig:
    learning_rate = 0.001
    batch_size = 32
    epochs = 100

class ModelConfig:
    architecture = "resnet50"
    hidden_size = 768

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Pass class objects directly
    experiment.params.log(training=TrainingConfig, model=ModelConfig)

    # Stored as:
    # training.learning_rate = 0.001
    # training.batch_size = 32
    # training.epochs = 100
    # model.architecture = "resnet50"
    # model.hidden_size = 768
```

Private attributes (starting with `_`) are automatically skipped.

## Updating Parameters

Call `parameters().set()` multiple times - values merge and overwrite:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Initial parameters
    experiment.params.set(learning_rate=0.001, batch_size=32)

    # Add more
    experiment.params.set(optimizer="adam", momentum=0.9)

    # Update existing
    experiment.params.set(learning_rate=0.0001)

    # Final result:
    # learning_rate = 0.0001  (updated)
    # batch_size = 32
    # optimizer = "adam"
    # momentum = 0.9
```

## Loading from Config Files

**From JSON:**

```{code-block} python
:linenos:

import json
from ml_dash import Experiment

with open("config.json", "r") as f:
    config = json.load(f)

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(**config)
    experiment.log("Configuration loaded")
```

**From command line arguments:**

```{code-block} python
:linenos:

import argparse
from ml_dash import Experiment

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(**vars(args))
```

**From dataclass:**

```{code-block} python
:linenos:

from dataclasses import dataclass, asdict
from ml_dash import Experiment

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

config = TrainingConfig()

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(**asdict(config))
```

**From params-proto (or any class):**

```{code-block} python
:linenos:

from ml_dash import Experiment

# Works with params_proto or any Python class
class Args:
    batch_size = 64
    learning_rate = 0.001

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Pass class directly - automatically extracts attributes
    experiment.params.log(Args=Args)
```

## Complete Training Configuration

```{code-block} python
:linenos:

with Experiment(name="resnet-imagenet", project="cv",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(**{
        "model": {
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 1000
        },
        "data": {
            "dataset": "imagenet",
            "train_split": 0.8,
            "num_workers": 4
        },
        "training": {
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 0.1,
            "optimizer": "sgd",
            "momentum": 0.9
        }
    })
```

## Retrieving Parameters

Get parameters during or after an experiment:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(learning_rate=0.001, batch_size=32)

    # Retrieve flattened parameters
    params = experiment.params.get()
    print(params)
    # → {"learning_rate": 0.001, "batch_size": 32}

    # Retrieve as nested dictionary
    params_nested = experiment.params.get(flatten=False)
    print(params_nested)
    # → {"learning_rate": 0.001, "batch_size": 32}
```

## API Methods

### `set()` / `log()`

Both methods do the same thing - set or merge parameters:

- **Nested dicts** are automatically flattened to dot notation
- **Class objects** are converted to dictionaries by extracting their attributes
- **Multiple calls** merge parameters (later values override earlier ones)
- Returns self for potential chaining

The `log()` method exists for semantic clarity but behaves identically to `set()`.

### `get()`

Retrieve current parameters:

- **`flatten=True` (default)**: Returns flattened dict with dot notation
- **`flatten=False`**: Returns nested dict structure

## Storage Format

**Local mode** - Stored as JSON:

```bash
cat ./experiments/project/my-experiment/parameters.json
```

```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "optimizer": "adam",
  "model.architecture": "resnet50",
  "model.pretrained": true
}
```

**Remote mode** - Stored in MongoDB as a document.

---

**Next:** Learn about [Metrics](metrics.md) for time-series metrics tracking.
