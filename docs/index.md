# ML-Dash Documentation

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Claude Code Plugin

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin:

```
/plugin marketplace add fortyfive-labs/ml-dash
/plugin install ml-dash@ml-dash
```

Once installed, ask questions like:

```console
$ claude "How do I log parameters from a config class?"
$ claude "Show me an example of tracking metrics during training"
```

## Installation

```{parsed-literal}
pip install ml-dash=={{version}}
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

## Documentation

### Core Documentation

- [Getting Started](getting-started.md)
- [Experiments](experiments.md)
- [Parameters](parameters.md)
- [Metrics](metrics.md)
- [Logging](logging.md)
- [Files](files.md)
- [CLI Commands](cli.md)
- [API Reference](api-reference.md)
- [Examples](complete-examples.md)

### Advanced Features

- **[Background Buffering](buffering.md)** - Non-blocking I/O with automatic batching
- **[Track API](tracks.md)** - Time-series data tracking for robotics & RL
- **[Image Saving](images.md)** - Direct numpy array to PNG/JPEG conversion

## Links

- **GitHub**: https://github.com/fortyfive-labs/ml-dash
- **PyPI**: https://pypi.org/project/ml-dash/
- **Dashboard**: https://dash.ml

```{toctree}
:maxdepth: 2
:caption: Core Documentation
:hidden:

getting-started
experiments
parameters
metrics
logging
files
cli
api-reference
complete-examples
```

```{toctree}
:maxdepth: 2
:caption: Advanced Features
:hidden:

buffering
tracks
images
```
