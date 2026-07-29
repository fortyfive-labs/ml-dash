# ML-Dash

[![PyPI version](https://img.shields.io/pypi/v/ml-dash.svg?style=flat&color=blue)](https://pypi.org/project/ml-dash/)

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Installation

```bash
$ pip install ml-dash
```

## Claude Code Plugin

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin:

```bash
$ /plugin marketplace add fortyfive-labs/ml-dash
$ /plugin install ml-dash@ml-dash
```

Once installed, ask questions like:

```bash
$ claude "How do I log parameters from a config class?"

You can log parameters using exp.params.set():

    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        model="resnet50"
    )

Or from a params-proto config: exp.params.update(Config)
```

```bash
$ claude "Show me an example of tracking metrics"

Here's a typical training loop with metrics:

    for epoch in range(100):
        loss = train_epoch()
        exp.metrics("train").log(loss=loss, epoch=epoch)

Metrics are automatically batched and synced in the background.
```

## Quick Start

```bash
from ml_dash import Experiment

# Local mode (no authentication required)
with Experiment(prefix="my-user/my-project/exp1", dash_root=".dash").run as exp:
    exp.logs.info("Training started")
    exp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = train_one_epoch()
        exp.metrics("train").log(loss=loss, epoch=epoch)
```

## Remote Mode (with dash.ml)

```bash
# Authenticate first
$ ml-dash login
```

```bash
from ml_dash import Experiment

with Experiment(
    prefix="my-user/my-project/exp1",
    dash_url="https://api.dash.ml"
).run as exp:
    exp.logs.info("Training on remote")
    exp.params.set(learning_rate=0.001)
```

## Documentation

### Core Documentation

- [Getting Started](reference/get-started-installation.md)
- [Experiments](reference/guides-experiments.md)
- [Parameters](reference/guides-parameters.md)
- [Metrics](reference/guides-metrics.md)
- [Logging](reference/guides-logging.md)
- [Files](reference/guides-files.md)
- [CLI Commands](reference/reference-cli.md)
- [API Reference](reference/reference-api.md)
- [Examples](reference/examples-complete.md)

### Dashboard

- **[Dashboard](reference/dashboard-overview.md)** - Web interface overview and experiment view
- **[Experiment Charts](reference/dashboard-charts.md)** - Configure charts with `.dashrc`
- **[Comparing Experiments](reference/dashboard-compare.md)** - Live Compare and Compare View
- **[`.dashrc` Reference](reference/dashboard-dashrc.md)** - Complete field reference

### Advanced Features

- **[Background Buffering](reference/guides-buffering.md)** - Non-blocking I/O with automatic batching
- **[Track API](reference/guides-tracks.md)** - Time-series data tracking for robotics & RL
- **[Image Saving](reference/guides-images.md)** - Direct numpy array to PNG/JPEG conversion

## Links

- **GitHub**: https://github.com/fortyfive-labs/ml-dash
- **PyPI**: https://pypi.org/project/ml-dash/
- **Dashboard**: https://dash.ml
