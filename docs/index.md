# ML-Dash

Simple, flexible ML experiment tracking.

## Claude Code Plugin

```bash
/plugin marketplace add fortyfive-labs/ml-dash
/plugin install ml-dash@ml-dash
```

## Installation

```shell
pip install ml-dash
# or: uv add ml-dash
```

## Quick Start

```python
from ml_dash import Experiment

with Experiment(prefix="alice/project/exp1", dash_root=".dash").run as exp:
    exp.log("Training started")
    exp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = train_one_epoch()
        exp.metrics("train").log(loss=loss, epoch=epoch)
```

## Remote Mode

```bash
ml-dash login  # Authenticate first
```

```python
with Experiment(prefix="alice/project/exp1", dash_url="https://api.dash.ml").run as exp:
    exp.log("Training on remote")
```

## CLI

```bash
ml-dash login                          # Authenticate
ml-dash create -p alice/my-project     # Create project
ml-dash upload --prefix alice/project  # Upload data
ml-dash list --prefix alice/project    # List experiments
```

## Documentation

- [Getting Started](getting-started.md)
- [Experiments](experiments.md) - RUN.entry, prefixes, lifecycle
- [Parameters](parameters.md)
- [Metrics](metrics.md)
- [Files](files.md)
- [Logging](logging.md)
- [Tracks](tracks.md) - Time-series data
- [Images](images.md) - Numpy to PNG/JPEG
- [Buffering](buffering.md) - Background I/O
- [CLI Reference](cli.md)
- [API Reference](api-reference.md)
- [Examples](complete-examples.md)

## Links

- [GitHub](https://github.com/fortyfive-labs/ml-dash)
- [PyPI](https://pypi.org/project/ml-dash/)
- [Dashboard](https://dash.ml)

```{toctree}
:maxdepth: 2
:hidden:

getting-started
experiments
parameters
metrics
files
logging
tracks
images
buffering
cli
api-reference
complete-examples
```
