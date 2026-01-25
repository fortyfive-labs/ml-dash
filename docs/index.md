# ML-Dash Documentation

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Claude Code Plugin

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin:

```
/plugin marketplace add fortyfive-labs/ml-dash
/plugin install ml-dash@ml-dash
```

Once installed, ask questions like:

<pre class="terminal">
<span class="prompt">$</span> claude <span class="string">"How do I log parameters from a config class?"</span>

<span class="response">❯</span> You can log parameters using <span class="func">exp.params.set()</span>:

    <span class="obj">exp</span>.<span class="method">params</span>.<span class="method">set</span>(
        <span class="param">learning_rate</span>=<span class="num">0.001</span>,
        <span class="param">batch_size</span>=<span class="num">32</span>,
        <span class="param">model</span>=<span class="string">"resnet50"</span>
    )

  Or from a params-proto config: <span class="func">exp.params.update(Config)</span>
</pre>

<pre class="terminal">
<span class="prompt">$</span> claude <span class="string">"Show me an example of tracking metrics"</span>

<span class="response">❯</span> Here's a typical training loop with metrics:

    <span class="keyword">for</span> epoch <span class="keyword">in</span> <span class="builtin">range</span>(<span class="num">100</span>):
        loss = train_epoch()
        <span class="obj">exp</span>.<span class="method">metrics</span>(<span class="string">"train"</span>).<span class="method">log</span>(<span class="param">loss</span>=loss, <span class="param">epoch</span>=epoch)

  <span class="dim">Metrics are automatically batched and synced in the background.</span>
</pre>

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
