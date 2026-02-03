# Quick Start Guide

This is a standard uv-based Python project with ML experiment examples.

## Project Structure

```
examples/
├── src/ml_experiments/     # Source package
│   ├── sweeps/            # Ad hoc experiments
│   └── baselines/         # Reference baselines
├── tests/                 # Test files
├── pyproject.toml         # Project metadata
└── uv.lock               # Dependency lock file
```

## Setup

```bash
# Clone and navigate to examples
cd /path/to/ml-dash/examples

# Install dependencies
uv sync

# Authenticate with ML-Dash
ml-dash login
```

## Running Experiments

### Ad Hoc Experiments (Sweeps)

```bash
# Navigate to sweeps directory
cd src/ml_experiments/sweeps

# Run a single training
uv run python train.py --train.learning-rate 0.01

# Run a sweep (multiple configs)
uv run python launch.py

# Run specific sweep file
uv run python launch.py --sweep configs/lr_sweep.jsonl

# Dry run (preview without executing)
uv run python launch.py --dry-run
```

### Baseline Experiments

```bash
# Navigate to baselines directory
cd src/ml_experiments/baselines

# Run a single baseline
uv run python train.py --train.learning-rate 0.01

# Run baseline sweep
uv run python launch.py

# Dry run
uv run python launch.py --dry-run
```

## Development

### Running Tests

```bash
# From project root (examples/)
uv run pytest tests/ -v
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

### Regenerating Sweep Configs

```bash
cd src/ml_experiments/sweeps
uv run python configs/sweep_gen.py
uv run python configs/lr_sweep_gen.py
uv run python configs/optimizer_sweep_gen.py
uv run python configs/batch_sweep_gen.py
```

## Key Features

- **Standard src/ layout**: Proper Python package structure
- **Editable install**: Changes to ml-dash parent package are immediately available
- **Type checking ready**: Configured with ruff for linting
- **Test suite**: pytest configured and ready to use
- **Reproducible**: uv.lock ensures consistent dependencies

## Tips

- Run commands from the examples/ root using `uv run python src/ml_experiments/sweeps/train.py`
- Or navigate to the script directory and run `uv run python train.py`
- Use `--dry-run` with launchers to preview what will execute
- Check the main README.md for detailed documentation

## Next Steps

See [README.md](README.md) for:
- Detailed feature documentation
- Multi-config setup with params-proto
- Sweep coordination and dashboard organization
- Advanced usage patterns
