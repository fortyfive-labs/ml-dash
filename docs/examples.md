# ML-Dash Examples

This directory contains runnable examples demonstrating ML-Dash features.

## Running Examples

All examples work with local mode (no server needed):

```bash
# Run from the project root with proper PYTHONPATH
PYTHONPATH=./src python docs/examples/three_usage_styles.py
PYTHONPATH=./src python docs/examples/01_basic_experiment.py
PYTHONPATH=./src python docs/examples/02_logging_example.py
PYTHONPATH=./src python docs/examples/03_parameters_example.py
PYTHONPATH=./src python docs/examples/04_metrics_example.py
PYTHONPATH=./src python docs/examples/05_files_example.py
PYTHONPATH=./src python docs/examples/06_complete_training.py
```

## Examples Overview

| Example | Description | Features |
|---------|-------------|----------|
| `three_usage_styles.py` | **NEW!** Three ways to use experiments | Decorator, context manager, direct instantiation |
| `01_basic_experiment.py` | Your first ML-Dash experiment | Experiment creation, logging, parameters |
| `02_logging_example.py` | Structured logging | Log levels, metadata, progress logging |
| `03_parameters_example.py` | Hyperparameters metricing | Simple params, nested params, updates |
| `04_metrics_example.py` | Time-series metrics | Metrics, batch append, reading data, stats |
| `05_files_example.py` | File uploads | Upload models, configs, results |
| `06_complete_training.py` | End-to-end training | All features combined |

## What Gets Created

All examples create data in `tutorial_data/.ml-dash/tutorials/` directory:

```
tutorial_data/.ml-dash/tutorials/
├── hello-ml-dash/
│   ├── logs.jsonl
│   └── parameters.json
├── logging-demo/
│   └── logs.jsonl
├── parameters-demo/
│   └── parameters.json
├── metrics-demo/
│   ├── logs.jsonl
│   ├── parameters.json
│   └── metrics/
│       ├── train_loss/
│       │   ├── data.jsonl
│       │   └── metadata.json
│       └── ...
└── ...
```

## Exploring Your Data

After running examples, explore the generated data:

```bash
# View logs
cat tutorial_data/.ml-dash/tutorials/hello-ml-dash/logs.jsonl

# View parameters
cat tutorial_data/.ml-dash/tutorials/parameters-demo/parameters.json

# View metric data
cat tutorial_data/.ml-dash/tutorials/metrics-demo/metrics/train_loss/data.jsonl

# List all experiments
ls tutorial_data/.ml-dash/tutorials/
```

## Clean Up

Remove all tutorial data:

```bash
rm -rf tutorial_data/
```

## Example Files

### Three Usage Styles (`three_usage_styles.py`)

Demonstrates all three ways to use ML-Dash experiments:
- **Decorator style**: Best for ML training functions
- **Context manager style**: Best for scripts
- **Direct instantiation**: Advanced usage with manual lifecycle management

### Basic Experiment (`01_basic_experiment.py`)

Your first ML-Dash experiment showing:
- Creating a local experiment
- Basic logging
- Setting parameters

### Logging Example (`02_logging_example.py`)

Comprehensive logging demonstration:
- Different log levels (debug, info, warn, error, fatal)
- Adding metadata to logs
- Progress logging
- Structured logging patterns

### Parameters Example (`03_parameters_example.py`)

Hyperparameter metricing:
- Simple key-value parameters
- Nested parameters with dot notation
- Updating parameters
- Parameter organization

### Metrics Example (`04_metrics_example.py`)

Time-series metrics metricing:
- Creating metrics
- Single and batch data appends
- Flexible schemas
- Reading metric data
- Getting statistics
- Listing all metrics

### Files Example (`05_files_example.py`)

File upload and management:
- Uploading files
- Adding descriptions and tags
- Using prefixes for organization
- Adding metadata to files
- Listing uploaded files

### Complete Training Example (`06_complete_training.py`)

End-to-end ML training simulation that combines all features:
- Experiment creation
- Parameter configuration
- Progress logging
- Metric metricing (loss, accuracy, learning rate)
- File uploads (checkpoints, final model, results)
- Complete workflow demonstration

## Next Steps

- Read the full tutorial documentation
- Explore the [API Quick Reference](api-quick-reference.md)
- Learn about specific features:
  - [Getting Started](getting-started.md)
  - [Experiments](experiments.md)
  - [Logging](logging.md)
  - [Parameters](parameters.md)
  - [Metrics](metrics.md)
  - [Files](files.md)
