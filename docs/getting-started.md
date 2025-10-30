# Getting Started with DreamLake

This guide will help you get started with DreamLake.

## Installation

```bash
# Install from source (for now)
cd ml-dash_python_sdk
pip install -e .
```

## Core Concepts

### Experiments

A **Experiment** represents a single experiment run or training experiment. Experiments contain:
- Logs (structured logging)
- Parameters (hyperparameters and configuration)
- Metrics (time-series metrics like loss, accuracy)
- Files (models, datasets, artifacts)

### Projects

A **Project** is a container for organizing related experiments. Think of it as a project or team project.

### Local vs Remote Mode

DreamLake operates in two modes:

- **Local Mode**: Data stored in filesystem (`.ml-dash/` directory)
- **Remote Mode**: Data stored in MongoDB + S3 via API

## Your First Experiment

DreamLake supports **three usage styles**. Choose the one that fits your workflow best:

### Style 1: Decorator (Recommended for ML Training)

Perfect for wrapping training functions:

```python
from ml_dash import ml_dash_experiment

@ml_dash_experiment(
    name="hello-ml-dash",
    project="tutorials",
    local_prefix="./my_experiments"
)
def my_first_experiment(experiment):
    """Experiment is automatically injected as a parameter"""
    # Log a message
    experiment.log("Hello from DreamLake!", level="info")

    # Metric a parameter
    experiment.parameters().set(message="Hello World")

    print("Experiment created successfully!")
    return "Done!"

# Run the experiment - experiment is managed automatically
result = my_first_experiment()
```

### Style 2: Context Manager (Recommended for Scripts)

The most common and Pythonic approach:

```python
from ml_dash import Experiment

# Create a experiment in local mode
with Experiment(
    name="hello-ml-dash",
    project="tutorials",
    local_prefix="./my_experiments",
        local_path=".ml-dash"
) as experiment:
    # Log a message
    experiment.log("Hello from DreamLake!", level="info")

    # Metric a parameter
    experiment.parameters().set(message="Hello World")

    print("Experiment created successfully!")
    print(f"Data stored in: {experiment._storage.root_path}")
```

### Style 3: Direct Instantiation (Advanced)

For fine-grained control:

```python
from ml_dash import Experiment

# Create an experiment
experiment = Experiment(
    name="hello-ml-dash",
    project="tutorials",
    local_prefix="./my_experiments",
        local_path=".ml-dash"
)

# Explicitly open
experiment.open()

try:
    # Log a message
    experiment.log("Hello from DreamLake!", level="info")

    # Metric a parameter
    experiment.parameters().set(message="Hello World")

    print("Experiment created successfully!")
finally:
    # Explicitly close
    experiment.close()
```

Save this as `hello_ml-dash.py` and run it:

```bash
python hello_ml-dash.py
```

You should see:
```
Experiment created successfully!
Data stored in: ./my_experiments
```

## What Just Happened?

1. **Experiment Created**: A new experiment named "hello-ml-dash" was created in the "tutorials" project
2. **Log Written**: A log message was written to `.ml-dash/tutorials/hello-ml-dash/logs.jsonl`
3. **Parameter Saved**: The parameter was saved to `.ml-dash/tutorials/hello-ml-dash/parameters.json`
4. **Auto-Closed**: The `with` statement automatically closed the experiment

## Inspecting Your Data

Let's check what was created:

```bash
# View the directory structure
tree ./my_experiments/.ml-dash

# View logs
cat ./my_experiments/.ml-dash/tutorials/hello-ml-dash/logs.jsonl

# View parameters
cat ./my_experiments/.ml-dash/tutorials/hello-ml-dash/parameters.json
```

## Experiment Context Manager

DreamLake uses Python's context manager pattern (`with` statement) to ensure proper cleanup:

```python
# ✓ Good - Automatic cleanup
with Experiment(name="my-experiment", project="test", local_prefix="./data",
        local_path=".ml-dash") as experiment:
    experiment.log("Training started")
    # ... do work ...
# Experiment automatically closed here

# ✗ Manual cleanup (not recommended)
experiment = Experiment(name="my-experiment", project="test", local_prefix="./data",
        local_path=".ml-dash")
experiment.open()
try:
    experiment.log("Training started")
finally:
    experiment.close()
```

## Experiment Metadata

You can add metadata to your experiments:

```python
with Experiment(
    name="mnist-baseline",
    project="computer-vision",
    local_prefix="./experiments",
    description="Baseline CNN for MNIST classification",
    tags=["mnist", "cnn", "baseline"],
    folder="/experiments/mnist",
        local_path=".ml-dash"
) as experiment:
    experiment.log("Experiment created with metadata")
```

## Error Handling

Experiments handle errors gracefully:

```python
from ml_dash import Experiment

try:
    with Experiment(
        name="test-experiment",
        project="test",
        local_prefix="./data",
        local_path=".ml-dash"
    ) as experiment:
        experiment.log("Starting work...")
        # Your code here
        raise Exception("Something went wrong!")
except Exception as e:
    print(f"Error occurred: {e}")
    # Experiment is still properly closed
```

## Next Steps

Now that you understand the basics, explore:
- [Experiments](experiments.md) - Advanced experiment management
- [Logging](logging.md) - Structured logging
- [Parameters](parameters.md) - Parameter metricing
- [Metrics](metrics.md) - Time-series metrics
- [Files](files.md) - File uploads

## Quick Reference

### Three Usage Styles

```python
from ml_dash import Experiment, ml_dash_experiment

# ========================================
# Style 1: Decorator (ML Training)
# ========================================
@ml_dash_experiment(
    name="experiment-name",
    project="project-name",
    local_prefix="./path/to/data"
)
def train(experiment):
    experiment.log("Training...")

train()  # Experiment managed automatically

# ========================================
# Style 2: Context Manager (Scripts)
# ========================================
# Local mode (filesystem)
with Experiment(
    name="experiment-name",
    project="project-name",
    local_prefix="./path/to/data",
        local_path=".ml-dash"
) as experiment:
    pass

# Remote mode (API + S3) - with username
with Experiment(
    name="experiment-name",
    project="project-name",
    remote="https://cu3thurmv3.us-east-1.awsapprunner.com",
    user_name="your-username"
) as experiment:
    pass

# Remote mode (API + S3) - with API key (advanced)
with Experiment(
    name="experiment-name",
    project="project-name",
    remote="https://cu3thurmv3.us-east-1.awsapprunner.com",
    api_key="your-api-key"
) as experiment:
    pass

# ========================================
# Style 3: Direct Instantiation (Advanced)
# ========================================
experiment = Experiment(
    name="experiment-name",
    project="project-name",
    local_prefix="./path/to/data",
        local_path=".ml-dash"
)
experiment.open()
try:
    # Do work
    pass
finally:
    experiment.close()
```

### All Styles Work With Remote Mode

```python
# Decorator + Remote
@ml_dash_experiment(
    name="experiment-name",
    project="project-name",
    remote="https://cu3thurmv3.us-east-1.awsapprunner.com",
    user_name="your-username"
)
def train(experiment):
    pass
```

**Note**: Using `user_name` is simpler for development - it automatically generates an API key from your username.

---

## See Also

Now that you know the basics, explore these guides:

- **[Architecture](architecture.md)** - Understand how DreamLake works internally
- **[Deployment Guide](deployment.md)** - Deploy your own DreamLake server
- **[API Quick Reference](api-quick-reference.md)** - Cheat sheet for common patterns
- **[Complete Examples](complete-examples.md)** - End-to-end ML workflows
- **[FAQ & Troubleshooting](faq.md)** - Common questions and solutions

**Feature-specific guides:**
- [Experiments](experiments.md) - Experiment lifecycle and management
- [Logging](logging.md) - Structured logging with levels
- [Parameters](parameters.md) - Hyperparameter metricing
- [Metrics](metrics.md) - Time-series metrics
- [Files](files.md) - File upload and management
