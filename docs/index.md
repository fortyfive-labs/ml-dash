# ML-Dash

[![PyPI version](https://img.shields.io/pypi/v/ml-dash.svg?style=flat&color=blue)](https://pypi.org/project/ml-dash/)

ML-Dash is a simple and flexible SDK for ML experiment tracking and data storage.

## Installation

<pre class="terminal"><span class="prompt">$</span> pip install ml-dash</pre>

## Claude Code Plugin

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin:

<pre class="terminal"><span class="prompt">$</span> /plugin marketplace add fortyfive-labs/ml-dash
<span class="prompt">$</span> /plugin install ml-dash@ml-dash</pre>

Once installed, ask questions like:

<pre class="terminal"><span class="prompt">$</span> claude <span class="string">"How do I log parameters from a config class?"</span>

You can log parameters using <span class="func">exp.params.set()</span>:

    <span class="obj">exp</span>.<span class="method">params</span>.<span class="method">set</span>(
        <span class="param">learning_rate</span>=<span class="num">0.001</span>,
        <span class="param">batch_size</span>=<span class="num">32</span>,
        <span class="param">model</span>=<span class="string">"resnet50"</span>
    )

Or from a params-proto config: <span class="func">exp.params.update(Config)</span></pre>

<pre class="terminal"><span class="prompt">$</span> claude <span class="string">"Show me an example of tracking metrics"</span>

Here's a typical training loop with metrics:

    <span class="keyword">for</span> epoch <span class="keyword">in</span> <span class="builtin">range</span>(<span class="num">100</span>):
        loss = train_epoch()
        <span class="obj">exp</span>.<span class="method">metrics</span>(<span class="string">"train"</span>).<span class="method">log</span>(<span class="param">loss</span>=loss, <span class="param">epoch</span>=epoch)

<span class="dim">Metrics are automatically batched and synced in the background.</span></pre>

## Quick Start

<pre class="terminal"><span class="keyword">from</span> ml_dash <span class="keyword">import</span> <span class="obj">Experiment</span>

<span class="comment"># Local mode (no authentication required)</span>
<span class="keyword">with</span> <span class="obj">Experiment</span>(<span class="param">prefix</span>=<span class="string">"my-user/my-project/exp1"</span>, <span class="param">dash_root</span>=<span class="string">".dash"</span>).run <span class="keyword">as</span> exp:
    exp.<span class="method">logs</span>.<span class="method">info</span>(<span class="string">"Training started"</span>)
    exp.<span class="method">params</span>.<span class="method">set</span>(<span class="param">learning_rate</span>=<span class="num">0.001</span>, <span class="param">batch_size</span>=<span class="num">32</span>)

    <span class="keyword">for</span> epoch <span class="keyword">in</span> <span class="builtin">range</span>(<span class="num">10</span>):
        loss = train_one_epoch()
        exp.<span class="method">metrics</span>(<span class="string">"train"</span>).<span class="method">log</span>(<span class="param">loss</span>=loss, <span class="param">epoch</span>=epoch)</pre>

## Remote Mode (with dash.ml)

<pre class="terminal"><span class="comment"># Authenticate first</span>
<span class="prompt">$</span> ml-dash login</pre>

<pre class="terminal"><span class="keyword">from</span> ml_dash <span class="keyword">import</span> <span class="obj">Experiment</span>

<span class="keyword">with</span> <span class="obj">Experiment</span>(
    <span class="param">prefix</span>=<span class="string">"my-user/my-project/exp1"</span>,
    <span class="param">dash_url</span>=<span class="string">"https://api.dash.ml"</span>
).run <span class="keyword">as</span> exp:
    exp.<span class="method">logs</span>.<span class="method">info</span>(<span class="string">"Training on remote"</span>)
    exp.<span class="method">params</span>.<span class="method">set</span>(<span class="param">learning_rate</span>=<span class="num">0.001</span>)</pre>

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
