# Getting Started

## Installation

To install ML-Logger, use pip:

```bash
pip install ml-dash
```

### Configuration

You can configure ML-Logger using environment variables or explicit configuration:

**Option 1: Environment Variables**

```bash
# Configure using environment variables
export ML_DASH_NAMESPACE="your-username"
export ML_DASH_WORKSPACE="your-project"
export ML_DASH_REMOTE="http://localhost:3001"  # Optional: ML-Dash server URL
```

When using `ml_dash.autolog`, these environment variables are automatically used:

```python
from ml_dash.autolog import experiment

# Automatically configured from environment variables
experiment.params.set(learning_rate=0.001)
```

**Option 2: Explicit Configuration**

```python
from ml_dash import Experiment

experiment = Experiment(
    namespace="your-username",
    workspace="your-project",
    prefix="experiment-1",
    remote="http://localhost:3001",  # Optional remote server
)
```


## Basic Usage

### Initialize an Experiment

```python
from ml_dash import Experiment

# Create an experiment for a single training execution
experiment = Experiment(
    namespace="your-username",
    workspace="project-name",
    prefix="experiments/my-experiment",
    readme="Testing ResNet50 with different learning rates"  # Searchable
)
```

Or use auto-configuration for zero-boilerplate logging:

```python
from ml_dash.autolog import experiment

# No setup needed! Auto-configured from environment variables
experiment.params.set(learning_rate=0.001)
experiment.readme = "Quick experiment"
```

### Log Parameters

Log hyperparameters and configuration at the start of your experiment:

```python
# Set parameters (replaces existing)
experiment.params.set(
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam",
    model="resnet50",
    dataset="imagenet",
    # Nested parameters for better organization
    train=dict(
        epochs=100,
        early_stopping=True,
    ),
    model=dict(
        layers=50,
        pretrained=True,
    )
)

# Or extend existing parameters (deep merge)
experiment.params.extend(
    train=dict(warmup_steps=1000)  # Adds to existing train dict
)

# Update a single parameter
experiment.params.update("learning_rate", 0.0001)
```

### Log Metrics

Log training and validation metrics during your training loop:

```python
for epoch in range(100):
    # Training
    train_loss = train_epoch()

    # Validation
    val_loss, val_acc = validate()

    # Log metrics immediately
    experiment.metrics.log(
        step=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_accuracy=val_acc,
    )
```

### Collecting Statistics

Sometimes we want to collect averaged statistics over multiple steps. You can do so via:

```python
# Collect statistics over mini-batches within an epoch
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(model, data, target)

        # Store metrics for averaging (doesn't log yet)
        experiment.metrics.collect(
            train_loss=loss,
            learning_rate=optimizer.param_groups[0]['lr'],
        )

    # Validation metrics logged at a lower frequency
    val_loss, val_acc = validate()

    # At end of epoch, log averaged statistics
    experiment.metrics.flush(
        # Compute mean over stored values
        _aggregation="mean",
        # Log with epoch as the step
        step=epoch,
        # Can also add custom metrics that aren't aggregated
        val_loss=val_loss,
        val_accuracy=val_acc,
    )
```

This approach is useful for reducing logging overhead while still capturing detailed statistics.

### Using Namespaced Metrics

Use namespaces to organize metrics into groups (e.g., train/eval, different experiments):

```python
# Create namespaced loggers
train_metrics = experiment.metrics("train")
eval_metrics = experiment.metrics("eval")

for epoch in range(100):
    # Training metrics
    train_metrics.log(step=epoch, loss=0.5, accuracy=0.8)

    # Evaluation metrics
    eval_metrics.log(step=epoch, loss=0.3, accuracy=0.9)

# Results in metrics: train/loss, train/accuracy, eval/loss, eval/accuracy
```

### Saving and Loading Artifacts

Save models, plots, and other files:

```python
# Save a model checkpoint
experiment.files.save(model.state_dict(), "model_epoch_10.pt")

# Save arbitrary data
experiment.files.save_pkl(data, "experiment_data.pkl")
experiment.files.save({"results": metrics}, "results.json")

# Use namespaced file storage
checkpoints = experiment.files("checkpoints")
checkpoints.save(model.state_dict(), "model_epoch_10.pt")
# Saves to: .ml-dash/.../files/checkpoints/model_epoch_10.pt
```

## Advanced Features

### Hierarchical Organization

Use prefixes to organize experiments hierarchically:

```python
# Organize by project, then experiment
experiment = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="object-detection/yolov8/lr_0.001"
)

# Or by date and time
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment = Experiment("alice", "experiments", f"runs/{timestamp}")
```

### Using experiment.run() - Three Patterns

The `experiment.run()` method supports three patterns for managing experiment lifecycle:

**Pattern 1: Direct call**
```python
experiment.run()  # Mark as started

for epoch in range(100):
    loss = train_epoch()
    experiment.metrics.log(step=epoch, loss=loss)

experiment.complete()  # Mark as completed
```

**Pattern 2: Context manager**
```python
with experiment.run():
    experiment.params.set(learning_rate=0.001)

    for epoch in range(100):
        loss = train_epoch()
        experiment.metrics.log(step=epoch, loss=loss)

    # Automatically calls experiment.complete() on success
    # Or experiment.fail() if exception occurs
```

**Pattern 3: Decorator**
```python
@experiment.run
def train(config):
    experiment.params.set(**config)

    for epoch in range(100):
        loss = train_epoch()
        experiment.metrics.log(step=epoch, loss=loss)

    return final_metrics

result = train(config)  # Auto-completes on success
```

### Local-First Logging Mode with Background Sync

ML-Logger writes everything to local files first for zero-latency logging. When a remote server is configured, a background daemon syncs the data:

```python
from ml_dash import Experiment

# Configure remote server
experiment = Experiment(
    namespace="alice",
    workspace="project",
    prefix="experiment-1",
    remote="http://localhost:3001",  # ML-Dash server
    local_root=".ml-dash"  # Local storage
)

# All operations write locally first, then sync in background
experiment.params.set(learning_rate=0.001)
experiment.metrics.log(step=0, loss=0.5)

# Files are written to .ml-dash/alice/project/experiment-1/
# Daemon automatically syncs *.jsonl files to remote server
```

## Viewing Results

View your logged experiments using the ML-Dash dashboard:

```bash
# Start the dashboard
ml-dash --data-dir ./experiments
```

Then navigate to `http://localhost:3000` to visualize your experiments.

## Developing ML-Logger

If you want to develop ML-Logger, you can install it in editable mode:

```bash
cd ml-dash
pip install -e '.[dev]'
```

To build the documentation:

```bash
make docs
```

## Next Steps

- Check out the [API Documentation](api/ml_dash.md) for detailed reference
- See the [CHANGE LOG](CHANGE_LOG.md) for version history
- Report issues on [GitHub](https://github.com/fortyfive-labs/ml-dash/issues)
