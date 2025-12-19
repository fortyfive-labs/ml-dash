# ML-Dash

A simple and flexible SDK for ML experiment tracking and data storage.

## Features

- **Three Usage Styles**: Pre-configured singleton (dxp), context manager, or direct instantiation
- **Dual Operation Modes**: Remote (API server) or local (filesystem)
- **OAuth2 Authentication**: Secure device flow authentication for CLI and SDK
- **Auto-creation**: Automatically creates namespace, project, and folder hierarchy
- **Upsert Behavior**: Updates existing experiments or creates new ones
- **Experiment Lifecycle**: Automatic status tracking (RUNNING, COMPLETED, FAILED, CANCELLED)
- **Organized File Storage**: Prefix-based file organization with unique snowflake IDs
- **Rich Metadata**: Tags, bindrs, descriptions, and custom metadata support
- **Simple API**: Minimal configuration, maximum flexibility

## Installation

<table>
<tr>
<td>Using uv (recommended)</td>
<td>Using pip</td>
</tr>
<tr>
<td>

```bash
uv add ml-dash==0.6.2rc1
```

</td>
<td>

```bash
pip install ml-dash==0.6.2rc1
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
from ml_dash import dxp

# Start experiment (uploads to https://api.dash.ml by default)
with dxp.run:
    dxp.log().info("Training started")
    dxp.params.set(learning_rate=0.001, batch_size=32)

    for epoch in range(10):
        loss = train_one_epoch()
        dxp.metrics("loss").append(value=loss, epoch=epoch)
```

#### Option B: Create Your Own Experiment

```python
from ml_dash import Experiment

with Experiment(
    name="my-experiment",
    project="my-project",
    remote="https://api.dash.ml"  # token auto-loaded
).run as experiment:
    experiment.log().info("Hello!")
    experiment.params.set(lr=0.001)
```

#### Option C: Local Mode (No Authentication Required)

```python
from ml_dash import Experiment

with Experiment(
    name="my-experiment",
    project="my-project",
    local_path=".ml-dash"
).run as experiment:
    experiment.log().info("Running locally")
```

See [docs/getting-started.md](docs/getting-started.md) for more examples.

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
