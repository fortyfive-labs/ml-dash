# Welcome to ML-Dash

## Installation

```shell
uv add ml-dash
```

or using pip

```shell
pip install ml-dash
```

## Quick Example

```python
from ml_dash import Experiment

with Experiment(name="my-experiment", project="my-project", local_path=".ml_dash") as experiment:
    # Log messages
    experiment.log("Training started")

    # Metric parameters
    experiment.parameters().set(learning_rate=0.001, batch_size=32)

    # Metric metrics
    experiment.metric("loss").append(value=0.5, epoch=1)
```

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

overview
quickstart
getting-started
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

experiments
logging
parameters
metrics
files
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples
basic-training
hyperparameter-search
model-comparison
complete-examples
```

```{toctree}
:maxdepth: 2
:caption: Advanced Topics
:hidden:

local-vs-remote
deployment
architecture
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api-quick-reference
api/modules
```

```{toctree}
:maxdepth: 1
:caption: Help
:hidden:

faq
```
