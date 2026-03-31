---
---

# ML-Dash

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Installation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::tip TIPS

We strongly recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment and dependencies. It is significantly faster than pip and provides reliable, reproducible installs.
:::

<Tabs>
  <TabItem value="pip" label="pip" default>
    ```bash
    pip install ml-dash
    ```
  </TabItem>
  <TabItem value="uv" label="uv">
    ```bash
    uv add ml-dash
    ```
  </TabItem>
  <TabItem value="conda" label="conda">
    ```bash
    conda install -c conda-forge ml-dash
    ```
  </TabItem>
</Tabs>



## Claude Code Plugin

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin:

```bash
/plugin marketplace add fortyfive-labs/ml-dash
/plugin install ml-dash@ml-dash
```

Once installed, ask questions like:

```python
claude "How do I log parameters from a config class?"

You can log parameters using exp.params.set():

    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        model="resnet50"
    )

Or from a params-proto config: exp.params.update(Config)
```

```python
claude "Show me an example of tracking metrics"

Here's a typical training loop with metrics:

    for epoch in range(100):
        loss = train_epoch()
        exp.metrics("train").log(loss=loss, epoch=epoch)

Metrics are automatically batched and synced in the background.
```

## Quick Start

```python
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

```python
# Authenticate first
ml-dash login
```

```python
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




