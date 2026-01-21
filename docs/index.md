# ML-Dash Documentation

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Installation

```shell
uv add ml-dash
```

or

```shell
pip install ml-dash
```

## Quick Start

```python
from ml_dash import Experiment

# Local mode (no authentication required)
with Experiment(prefix="my-user/my-project/exp1", dash_root=".dash").run as exp:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = train_one_epoch()
        exp.metrics("train").log(loss=loss, epoch=epoch)
```

## Remote Mode (with dash.ml)

```bash
# Authenticate first
ml-dash login

# Then use remote mode
```

```python
from ml_dash import Experiment

with Experiment(
    prefix="my-user/my-project/exp1",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.log("Training on remote")
    exp.params.set(learning_rate=0.001)
```

## Pre-configured Singleton

```python
from ml_dash.auto_start import dxp

with dxp.run:
    dxp.log("Using pre-configured experiment")
    dxp.params.set(learning_rate=0.001)
```

## CLI Commands

### Authentication
```bash
ml-dash login          # Login to dash.ml
ml-dash logout         # Logout
ml-dash profile        # Show current user
```

### Project Management
```bash
# Create a project in current user's namespace
ml-dash create -p new-project

# Create a project in a specific namespace
ml-dash create -p geyang/tutorials

# Create with description
ml-dash create -p geyang/tutorials -d "ML tutorials and examples"
```

### Data Operations
```bash
ml-dash upload --prefix my-user/my-project
ml-dash download --prefix my-user/my-project
ml-dash list --prefix my-user/my-project
```

## Documentation

The documentation is being reorganized. Current documentation can be found in the [archived](archived/) folder:

- [Getting Started](archived/getting-started.md)
- [Experiments](archived/experiments.md)
- [Parameters](archived/parameters.md)
- [Metrics](archived/metrics.md)
- [Logging](archived/logging.md)
- [Files](archived/files.md)
- [API Reference](archived/api-reference.md)
- [CLI Commands](archived/cli.md)
- [Examples](archived/complete-examples.md)

## Links

- **GitHub**: https://github.com/fortyfive-labs/ml-dash
- **PyPI**: https://pypi.org/project/ml-dash/
- **Dashboard**: https://dash.ml

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

archived/getting-started
archived/experiments
archived/parameters
archived/metrics
archived/logging
archived/files
archived/api-reference
archived/cli
archived/basic-training
archived/hyperparameter-search
archived/model-comparison
archived/complete-examples
```
