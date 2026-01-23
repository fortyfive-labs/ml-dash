# ML-Dash

A simple and flexible SDK for ML experiment tracking and data storage with background buffering for high-performance training.

## Features

### Core Features
- **Three Usage Styles**: Pre-configured singleton (dxp), context manager, or direct instantiation
- **Dual Operation Modes**: Remote (API server) or local (filesystem)
- **OAuth2 Authentication**: Secure device flow authentication for CLI and SDK
- **Auto-creation**: Automatically creates namespace, project, and folder hierarchy
- **Upsert Behavior**: Updates existing experiments or creates new ones
- **Experiment Lifecycle**: Automatic status tracking (RUNNING, COMPLETED, FAILED, CANCELLED)
- **Organized File Storage**: Prefix-based file organization with unique snowflake IDs
- **Rich Metadata**: Tags, bindrs, descriptions, and custom metadata support
- **Simple API**: Minimal configuration, maximum flexibility

### Performance Features (New in 0.6.7)
- **Background Buffering**: Non-blocking I/O operations eliminate training interruptions
- **Automatic Batching**: Time-based (5s) and size-based (100 items) flush triggers
- **Track API**: Time-series data tracking for robotics, RL, and sequential experiments
- **Numpy Image Support**: Direct saving of numpy arrays as PNG/JPEG images
- **Parallel Uploads**: ThreadPoolExecutor for efficient file uploads

## Installation

<table>
<tr>
<td>Using uv (recommended)</td>
<td>Using pip</td>
</tr>
<tr>
<td>

```bash
uv add ml-dash
```

</td>
<td>

```bash
pip install ml-dash
```

</td>
</tr>
</table>

## Quick Start

### 1. Authenticate (Required for Remote Mode)

```bash
ml-dash login
```

This opens your browser for secure OAuth2 authentication. Your credentials are stored securely in your system keychain.

### 2. Start Tracking Experiments

#### Option A: Use the Pre-configured Singleton (Easiest)

```python
from ml_dash.auto_start import dxp

# Start experiment (uploads to https://api.dash.ml by default)
with dxp.run:
    dxp.log("Training started", level="info")
    dxp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = train_one_epoch()
        dxp.metrics("train").log(loss=loss, epoch=epoch)
```

#### Option B: Create Your Own Experiment

```python
from ml_dash import Experiment

with Experiment(
  prefix="alice/my-project/my-experiment",
  dash_url="https://api.dash.ml",  # token auto-loaded
).run as experiment:
  experiment.log("Hello!", level="info")
  experiment.params.set(lr=0.001)
```

#### Option C: Local Mode (No Authentication Required)

```python
from ml_dash import Experiment

with Experiment(
  project="my-project", prefix="my-experiment", dash_root=".dash"
).run as experiment:
  experiment.log("Running locally", level="info")

```

## New Features in 0.6.7

### 🚀 Background Buffering (Non-blocking I/O)

All write operations are now buffered and executed in background threads:

```python
with Experiment("my-project/exp").run as experiment:
    for i in range(10000):
        # Non-blocking! Returns immediately
        experiment.log(f"Step {i}")
        experiment.metrics("train").log(loss=loss, accuracy=acc)
        experiment.files("frames").save_image(frame, to=f"frame_{i}.jpg")

    # All data automatically flushed when context exits
```

Configure buffering via environment variables:
```bash
export ML_DASH_BUFFER_ENABLED=true
export ML_DASH_FLUSH_INTERVAL=5.0
export ML_DASH_LOG_BATCH_SIZE=100
```

### 📊 Track API (Time-Series Data)

Perfect for robotics, RL, and sequential experiments:

```python
with Experiment("robotics/training").run as experiment:
    for step in range(1000):
        # Track robot position over time
        experiment.track("robot/position").append({
            "step": step,
            "x": position[0],
            "y": position[1],
            "z": position[2]
        })

        # Track control signals
        experiment.track("robot/control").append({
            "step": step,
            "motor1": ctrl[0],
            "motor2": ctrl[1]
        })
```

### 🖼️ Numpy Image Support

Save numpy arrays directly as images (PNG/JPEG):

```python
import numpy as np

with Experiment("vision/training").run as experiment:
    # From MuJoCo, OpenCV, PIL, etc.
    pixels = renderer.render()  # numpy array

    # Save as PNG (lossless)
    experiment.files("frames").save_image(pixels, to="frame.png")

    # Save as JPEG with quality control
    experiment.files("frames").save_image(pixels, to="frame.jpg", quality=85)

    # Auto-detection also works
    experiment.files("frames").save(pixels, to="frame.jpg")
```

See [CHANGELOG.md](CHANGELOG.md) for complete release notes.

## Development Setup

### Installing Dev Dependencies

To contribute to ML-Dash or run tests, install the development dependencies:

<table>
<tr>
<td>Using uv (recommended)</td>
<td>Using pip</td>
</tr>
<tr>
<td>

```bash
uv sync --extra dev
```

</td>
<td>

```bash
pip install -e ".[dev]"
```

</td>
</tr>
</table>

This installs:
- `pytest>=8.0.0` - Testing framework
- `pytest-asyncio>=0.23.0` - Async test support
- `sphinx>=7.2.0` - Documentation builder
- `sphinx-rtd-theme>=2.0.0` - Read the Docs theme
- `sphinx-autobuild>=2024.0.0` - Live preview for documentation
- `myst-parser>=2.0.0` - Markdown support for Sphinx
- `ruff>=0.3.0` - Linter and formatter
- `mypy>=1.9.0` - Type checker

### Running Tests

<table>
<tr>
<td>Using uv</td>
<td>Using pytest directly</td>
</tr>
<tr>
<td>

```bash
uv run pytest
```

</td>
<td>

```bash
pytest
```

</td>
</tr>
</table>

### Building Documentation

Documentation is built using Sphinx with Read the Docs theme.

<table>
<tr>
<td>Build docs</td>
<td>Live preview</td>
<td>Clean build</td>
</tr>
<tr>
<td>

```bash
uv run python -m sphinx -b html docs docs/_build/html
```

</td>
<td>

```bash
uv run sphinx-autobuild docs docs/_build/html
```

</td>
<td>

```bash
rm -rf docs/_build
```

</td>
</tr>
</table>

The live preview command starts a local server and automatically rebuilds when files change.

Alternatively, you can use the Makefile from within the docs directory:

```bash
cd docs
make html          # Build HTML documentation
make clean         # Clean build files
```

For maintainers, to build and publish a new release: `uv build && uv publish`
