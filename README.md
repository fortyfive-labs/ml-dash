# ML-Dash

A simple and flexible SDK for ML experiment metricing and data storage.

## Features

- **Three Usage Styles**: Decorator, context manager, or direct instantiation
- **Dual Operation Modes**: Remote (API server) or local (filesystem)
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

## Getting Started

### Remote Mode (with API Server)

```python
from ml_dash import Experiment

with Experiment(
    name="my-experiment",
    project="my-project",
    remote="https://api.dash.ml",
    api_key="your-jwt-token"
) as experiment:
    print(f"Experiment ID: {experiment.id}")
```

### Local Mode (Filesystem)

```python
from ml_dash import Experiment

with Experiment(
    name="my-experiment",
    project="my-project",
    local_path=".ml-dash"
) as experiment:
    pass  # Your code here
```

See [examples/](examples/) for more complete examples.

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
