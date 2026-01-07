# ML-Dash API Reference

Complete API reference for ML-Dash Python SDK.

## Table of Contents

- [Experiment](#experiment)
  - [Constructor](#experiment-constructor)
  - [Lifecycle Management (run)](#lifecycle-management)
  - [Properties](#experiment-properties)
- [Parameters (params)](#parameters)
- [Logging](#logging)
- [Metrics](#metrics)
- [Files](#files)
- [Auto-Start (dxp)](#auto-start-dxp)
- [Complete Examples](#complete-examples)

---

## Experiment

The `Experiment` class is the main entry point for ML-Dash. It represents a single machine learning experiment run.

### Experiment Constructor

```python
Experiment(
    name: str,
    project: str,
    *,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    bindrs: Optional[List[str]] = None,
    folder: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # Mode configuration
    remote: Optional[str] = None,
    api_key: Optional[str] = None,
    user_name: Optional[str] = None,
    local_path: Optional[str] = None,
)
```

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique experiment name within the project |
| `project` | `str` | Project name to organize experiments |

#### Optional Metadata Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | `None` | Human-readable description of the experiment |
| `tags` | `List[str]` | `None` | Tags for categorization and search |
| `bindrs` | `List[str]` | `None` | Binders for advanced organization |
| `prefix` | `str` | `None` | Logical folder path for organization (e.g., "experiments/baseline") |
| `metadata` | `Dict[str, Any]` | `None` | Additional structured metadata |

#### Mode Configuration

**Local Mode** (filesystem storage):
```python
Experiment(
    prefix="my-experiment",
    project="my-project",
      # Required for local mode
)
```

**Remote Mode** (API + S3 storage):
```python
# Option 1: With username (auto-generates API key)
Experiment(
    prefix="my-experiment",
    project="my-project",
    remote="https://api.dash.ml",
    user_name="your-username"
)

# Option 2: Default remote mode (defaults to https://api.dash.ml)
Experiment(
    prefix="my-experiment",
    project="my-project"
)

# Option 3: Custom remote server
Experiment(
    prefix="my-experiment",
    project="my-project",
    remote="https://custom-server.com"
)
```

---

### Lifecycle Management

Experiments are managed through the `run` property, which returns a `RunManager` instance. The RunManager supports three usage patterns:

#### 1. Context Manager (Recommended)

Automatically starts and completes/fails the experiment:

```python
with Experiment(prefix="exp", project="proj").run as exp:
    exp.log("Training started")
    exp.params.set(lr=0.001)
    # Experiment automatically completes on successful exit
    # or fails if an exception occurs
```

#### 2. Decorator Pattern

Perfect for wrapping training functions:

```python
exp = Experiment(prefix="exp", project="proj")

@exp.run
def train_model(experiment):
    experiment.log("Training...")
    experiment.params.set(lr=0.001)
    return "done"

result = train_model()
```

#### 3. Manual Control

Explicit start/complete for fine-grained control:

```python
exp = Experiment(prefix="exp", project="proj")

exp.run.start()
try:
    exp.log("Training...")
    exp.params.set(lr=0.001)
    exp.run.complete()
except Exception:
    exp.run.fail()
```

#### RunManager Methods

| Method | Description | Status Set |
|--------|-------------|------------|
| `run.start()` | Start the experiment | `RUNNING` |
| `run.complete()` | Mark experiment as successfully completed | `COMPLETED` |
| `run.fail()` | Mark experiment as failed | `FAILED` |
| `run.cancel()` | Mark experiment as cancelled | `CANCELLED` |

---

### Experiment Properties

| Property | Type | Description |
|----------|------|-------------|
| `experiment.name` | `str` | Experiment name |
| `experiment.project` | `str` | Project name |
| `experiment.description` | `str` | Experiment description |
| `experiment.tags` | `List[str]` | Experiment tags |
| `experiment.bindrs` | `List[str]` | Experiment bindrs |
| `experiment.folder` | `str` | Folder path |
| `experiment.id` | `str` | Experiment ID (remote mode only, after start) |
| `experiment.data` | `dict` | Full experiment data (remote mode only, after start) |

---

## Parameters

Access experiment parameters through the `params` property (not a method).

### Setting Parameters

```python
# Set simple parameters
exp.params.set(
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Set nested parameters (auto-flattened)
exp.params.set(
    model={
        "architecture": "resnet50",
        "layers": 50,
        "pretrained": True
    },
    optimizer={
        "type": "adam",
        "lr": 0.001
    }
)

# Update specific nested parameters
exp.params.set(model={"lr": 0.0001})  # Only updates model.lr
```

### Getting Parameters

```python
# Get flattened parameters (dot notation)
params = exp.params.get()
# Returns: {"learning_rate": 0.001, "model.architecture": "resnet50", ...}

# Get nested parameters (hierarchical structure)
params = exp.params.get(flatten=False)
# Returns: {"learning_rate": 0.001, "model": {"architecture": "resnet50", ...}}
```

### Parameters API

| Method | Returns | Description |
|--------|---------|-------------|
| `params.set(**kwargs)` | `ParametersBuilder` | Set/merge parameters (supports nested dicts) |
| `params.get(flatten=True)` | `dict` | Get parameters (flattened or nested) |

**Note:** Parameters are automatically flattened for storage using dot notation:
- Input: `{"model": {"lr": 0.001}}`
- Stored as: `{"model.lr": 0.001}`

---

## Logging

Log messages with different severity levels and optional metadata.

### Basic Logging

```python
# Simple log
exp.log("Training started")

# Log with level
exp.log("Training started", level="info")
exp.log("Warning: Low GPU memory", level="warning")
exp.log("Error occurred", level="error")
```

### Logging with Metadata

```python
# Traditional style
exp.log(
    "Epoch completed",
    level="info",
    metadata={"epoch": 1, "loss": 0.5, "accuracy": 0.85}
)

# Or pass metadata as kwargs
exp.log("Epoch completed", level="info", epoch=1, loss=0.5, accuracy=0.85)
```

### Fluent Logging API

```python
# Fluent style (returns LogBuilder)
exp.log(metadata={"epoch": 1}).info("Training started")
exp.log().error("Failed to load data", error_code=500)
exp.log().warning("GPU memory low", memory_available="1GB")
```

### Log Levels

| Level | Description |
|-------|-------------|
| `debug` | Detailed diagnostic information |
| `info` | General informational messages (default) |
| `warning` | Warning messages for potential issues |
| `error` | Error messages for failures |

---

## Metrics

Track time-series metrics like loss, accuracy, etc.

### Basic Usage

Log train and eval metrics together:

```python
# Log all metrics for an epoch in one call
exp.metrics.log(
    epoch=epoch,
    train=dict(loss=train_loss, accuracy=train_acc),
    eval=dict(loss=val_loss, accuracy=val_acc)
)
```

### Prefix-Based Logging

Log metrics using namespace prefixes:

```python
# Log train metrics with prefix
exp.metrics("train").log(loss=0.5, accuracy=0.85)

# Log eval metrics with prefix
exp.metrics("eval").log(loss=0.6, accuracy=0.82)

# Set context and flush
exp.metrics.log(epoch=epoch).flush()
```

### Buffer API

For high-frequency logging (per-batch), use the buffer API:

```python
for batch in dataloader:
    loss = train_step(batch)
    acc = compute_accuracy(batch)

    # Buffer per-batch values (not written to disk yet)
    exp.metrics("train").buffer(loss=loss, accuracy=acc)

# At end of epoch, compute and log summary statistics
exp.metrics.buffer.log_summary()  # default: mean
exp.metrics.log(epoch=epoch).flush()
```

### Multiple Aggregations

```python
# Default (just mean)
exp.metrics.buffer.log_summary()

# Multiple aggregations
exp.metrics.buffer.log_summary("mean", "std", "min", "max", "count")

# Percentiles
exp.metrics.buffer.log_summary("p50", "p90", "p95", "p99")
```

### Reading Metrics

```python
# Read metric data
data = exp.metrics("train_loss").read(start_index=0, limit=100)
# Returns: {
#     "data": [...],
#     "startIndex": 0,
#     "endIndex": 99,
#     "total": 1000,
#     "hasMore": True
# }

# Get statistics
stats = exp.metrics("train_loss").stats()
# Returns: {
#     "count": 1000,
#     "firstValue": {...},
#     "lastValue": {...},
#     ...
# }
```

### Metrics API

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `metrics(prefix)` | `str` | `MetricBuilder` | Create/get metric builder with prefix |
| `metrics.log(**data)` | Flexible data fields | `MetricBuilder` | Log metric data point |
| `metrics("prefix").log(**data)` | Flexible data fields | `MetricBuilder` | Log with prefix |
| `metrics("prefix").buffer(**data)` | Flexible data fields | `None` | Buffer values for summary |
| `metrics.buffer.log_summary(*aggs)` | aggregation names | `None` | Log summary statistics |
| `metrics.flush()` | - | `None` | Flush pending metrics |
| `metrics("prefix").read(start_index, limit)` | int, int | `dict` | Read data points |
| `metrics("prefix").stats()` | - | `dict` | Get metric statistics |

---

## Files

Upload, download, and manage files associated with experiments.

### Basic File Upload

```python
# Upload a file
result = exp.files("models").save("./model.pt")

# Upload with metadata
result = exp.files("models").save(
    "./model.pt",
    description="Best model checkpoint",
    tags=["best", "checkpoint"],
    metadata={"epoch": 50, "accuracy": 0.95}
)
```

### Enhanced File Operations

#### Save JSON Objects

```python
# Save dict/object as JSON
config = {
    "model": "resnet50",
    "lr": 0.001,
    "batch_size": 32
}

result = exp.files("configs").save_json(config, to="config.json")
```

#### Save PyTorch Models

```python
import torch

# Save PyTorch model
model = torch.nn.Linear(10, 5)
result = exp.files("models").save_torch(model, to="model.pt")

# Save state dict
result = exp.files("models").save_torch(model.state_dict(), to="model.pth")
```

#### Save Pickle Objects

```python
# Save any Python object as pickle
data = {"results": [1, 2, 3], "metadata": {"version": "1.0"}}
result = exp.files("data").save_pkl(data, to="data.pkl")
```

#### Save Matplotlib Figures

```python
import matplotlib.pyplot as plt

# Create plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Sample Plot")

# Save current figure
result = exp.files("plots").save_fig(to="plot.png")

# Save specific figure with custom DPI
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
result = exp.files("plots").save_fig(
    fig=fig,
    to="plot.pdf",
    dpi=150,
    bbox_inches='tight'
)
```

#### Save Videos

```python
import numpy as np

# Create video frames (grayscale or RGB)
frames = [np.random.rand(480, 640) for _ in range(30)]

# Save as MP4 (default 20 FPS)
result = exp.files("videos").save_video(frames, to="output.mp4")

# Save with custom FPS
result = exp.files("videos").save_video(frames, to="output.mp4", fps=30)

# Save as GIF
result = exp.files("videos").save_video(frames, to="animation.gif")

# Save with custom codec and quality
result = exp.files("videos").save_video(
    frames,
    to="high_quality.mp4",
    fps=30,
    codec='libx264',
    quality=8
)
```

### File Organization with Bindrs

```python
# Upload file with bindrs for advanced organization
result = exp.files("models").save(
    "./model.pt",
    bindrs=["v1", "best", "production"],
    description="Production model v1"
)
```

### Listing Files

```python
# List all files
files = exp.files().list()

# List files by prefix
files = exp.files("models").list()

# List files by tags
files = exp.files(tags=["checkpoint"]).list()
```

### Downloading Files

```python
# Download to current directory with original filename
path = exp.files(file_id="123").download()

# Download to custom path
path = exp.files(file_id="123").download(dest_path="./my_model.pt")
```

### File Management

```python
# Update file metadata
result = exp.files(
    file_id="123",
    description="Updated description",
    tags=["new", "tags"],
    metadata={"version": "2.0"}
).update()

# Delete file (soft delete)
result = exp.files(file_id="123").delete()
```

### Files API

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `file(**kwargs)` | See below | `FileBuilder` | Create file builder |
| `save()` | - | `dict` | Upload file |
| `save_json(content, filename)` | `Any`, `str` | `dict` | Save JSON object as file |
| `save_torch(model, filename)` | `Any`, `str` | `dict` | Save PyTorch model as file |
| `save_pkl(content, filename)` | `Any`, `str` | `dict` | Save Python object as pickle file |
| `save_fig(fig, filename, **kwargs)` | `Optional[Any]`, `str`, `**kwargs` | `dict` | Save matplotlib figure as file |
| `save_video(frames, filename, fps, **kwargs)` | `Union[List, Any]`, `str`, `int`, `**kwargs` | `dict` | Save video frame stack as file |
| `list()` | - | `List[dict]` | List files |
| `download(dest_path)` | `Optional[str]` | `str` | Download file |
| `update()` | - | `dict` | Update file metadata |
| `delete()` | - | `dict` | Delete file |

**FileBuilder kwargs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Logical path/prefix (e.g., "models", "configs") |
| `description` | `str` | File description |
| `tags` | `List[str]` | File tags |
| `bindrs` | `List[str]` | File bindrs |
| `metadata` | `dict` | File metadata |
| `file_id` | `str` | File ID (for download/update/delete) |
| `dest_path` | `str` | Destination path (for download) |

---

## Auto-Start (dxp)

The `ml_dash.auto_start` module provides a pre-configured, auto-started experiment singleton named `dxp` for quick prototyping and scratch work.

### Overview

The `dxp` singleton is:
- **Pre-configured**: Name is "dxp", project is "scratch", storage is local (`.dash`)
- **Auto-started**: Ready to use immediately on import - no need to call `.run.start()`
- **Auto-cleanup**: Automatically completed on Python exit via `atexit` handler
- **Fully-featured**: Works exactly like a normal `Experiment` instance

### Import

```python
from ml_dash.auto_start import dxp
```

### Usage

#### Basic Usage

```python
from ml_dash.auto_start import dxp

# Ready to use immediately - already started!
dxp.log("Starting quick experiment")
dxp.params.set(lr=0.001, batch_size=32)

# Log metrics
dxp.metrics("train").log(loss=0.5, step=0)
dxp.metrics("train").log(loss=0.4, step=1)

# Upload files
dxp.files("models").save("model.pt")

# Automatically completed on Python exit
```

#### Interactive/Notebook Usage

Perfect for Jupyter notebooks and interactive Python sessions:

```python
from ml_dash.auto_start import dxp

# Quick parameter tracking
dxp.params.set(
    model="resnet50",
    lr=0.001,
    batch_size=32,
    optimizer="adam"
)

# Log training progress
for epoch in range(10):
    loss = train_epoch()
    dxp.metrics("train").log(loss=loss, epoch=epoch)
    dxp.log(f"Epoch {epoch} completed", loss=loss)

# Save results
dxp.files("results").save_json(results, to="results.json")
```

#### Prototyping Scripts

Great for quick experiments and throwaway scripts:

```python
from ml_dash.auto_start import dxp
import torch

# Track experiment
dxp.params.set(model_name="simple_cnn", dataset="mnist")
dxp.log("Training started")

# Train model
model = train_model()

# Save model
dxp.files("models").save_torch(model, to="model.pt")

# Log final metrics
dxp.metrics("train").log(accuracy=0.95, step=10)
dxp.log("Training completed")

# No cleanup needed - automatically handled!
```

### Configuration

The `dxp` singleton has a fixed configuration:

| Property | Value | Changeable |
|----------|-------|------------|
| Name | `"dxp"` | No (read-only) |
| Project | `"scratch"` | No (read-only) |
| Storage Mode | Local (`.dash`) | No (fixed at initialization) |
| Local Path | `".dash"` | No (fixed at initialization) |
| Parameters | Empty (initially) | Yes (during lifecycle) |
| Tags | Empty (initially) | No |
| Description | None | No |

### Lifecycle

Unlike normal experiments, `dxp` handles lifecycle automatically:

```python
from ml_dash.auto_start import dxp

# Already started - no need to call .run.start()
assert dxp._is_open  # True

# Use normally
dxp.log("Working...")
dxp.params.set(foo="bar")

# Automatically completed on Python exit
# No need to call .run.complete()
```

### Manual Lifecycle Control

You can still manually control the lifecycle if needed:

```python
from ml_dash.auto_start import dxp

# Close manually if needed
dxp.run.complete()

# Can reopen
dxp.run.start()

# Or mark as failed
dxp.run.fail()
```

### Comparison with Regular Experiment

| Feature | Regular Experiment | dxp Singleton |
|---------|-------------------|---------------|
| Import | `from ml_dash import Experiment` | `from ml_dash.auto_start import dxp` |
| Configuration | User-defined | Fixed (dxp/scratch/local) |
| Lifecycle | Manual or context manager | Auto-started, auto-completed |
| Use Case | Production experiments | Quick prototyping, notebooks |
| Storage | Local or remote | Local only (`.dash`) |
| Reusable | Create multiple instances | Single global instance |

### Best Practices

1. **Use for Prototyping**: `dxp` is perfect for quick experiments and scratch work
2. **Not for Production**: Use regular `Experiment` for production code
3. **Notebook Sessions**: Ideal for Jupyter notebooks and interactive sessions
4. **Single Session**: Best for single-run scripts; for multiple runs, use regular `Experiment`
5. **Local Development**: Great for local development and debugging

### Example: Complete Prototype

```python
from ml_dash.auto_start import dxp
import torch
import torch.nn as nn

# Setup
dxp.params.set(
    model="simple_cnn",
    lr=0.001,
    epochs=10,
    batch_size=64
)

dxp.log("Starting prototype experiment")

# Define and train model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 13 * 13, 10)
)

for epoch in range(10):
    loss = train_epoch(model)
    acc = evaluate(model)

    dxp.metrics("train").log(loss=loss, accuracy=acc, epoch=epoch)
    dxp.log(f"Epoch {epoch}", loss=loss, accuracy=acc)

# Save results
dxp.files("models").save_torch(model.state_dict(), to="final.pt")
dxp.log("Experiment completed!")

# Automatically saved on exit!
```

---

## Complete Examples

### Complete Training Workflow

```python
from ml_dash import Experiment

# Create experiment
with Experiment(
    prefix="resnet50-training",
    project="image-classification",
    description="ResNet50 on CIFAR-10",
    tags=["resnet", "cifar10", "baseline"],
    
).run as exp:

    # 1. Set hyperparameters
    exp.params.set(
        model={
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 10
        },
        training={
            "optimizer": "adam",
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 100
        }
    )

    # 2. Save configuration
    config = {
        "model": "resnet50",
        "dataset": "cifar10",
        "lr": 0.001
    }
    exp.files("configs").save_json(config, to="config.json")

    # 3. Training loop
    exp.log("Training started", level="info")

    for epoch in range(100):
        # Train one epoch
        train_loss, train_acc = train_epoch()  # Your training code
        val_loss, val_acc = validate()  # Your validation code

        # Log metrics
        exp.metrics("train").log(loss=train_loss, accuracy=train_acc)
        exp.metrics("eval").log(loss=val_loss, accuracy=val_acc)
        exp.metrics.log(epoch=epoch).flush()

        # Log progress
        if epoch % 10 == 0:
            exp.log(
                f"Epoch {epoch} completed",
                level="info",
                train_loss=train_loss,
                val_acc=val_acc
            )

        # Save checkpoints
        if val_acc > best_acc:
            exp.files("models").save_torch(
                model.state_dict(),
                to=f"checkpoint_epoch_{epoch}.pt"
            )
            exp.log(f"New best model saved at epoch {epoch}", level="info")

    # 4. Save final model
    exp.files("models").save_torch(model, to="final_model.pt")

    exp.log("Training completed", level="info")
```

### Hyperparameter Search

```python
from ml_dash import Experiment
import itertools

# Define search space
learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]

for lr, bs in itertools.product(learning_rates, batch_sizes):
    # Create experiment for each combination
    with Experiment(
        name=f"search-lr{lr}-bs{bs}",
        project="hyperparam-search",
        tags=["search", "grid"],
        
    ).run as exp:

        exp.params.set(
            learning_rate=lr,
            batch_size=bs,
            model="resnet18"
        )

        # Train with these hyperparameters
        final_acc = train(lr=lr, batch_size=bs)

        exp.metrics("train").log(accuracy=final_acc, step=1)
        exp.log(f"Search completed: lr={lr}, bs={bs}, acc={final_acc}")
```

### Decorator Pattern for Training Functions

```python
from ml_dash import Experiment

# Create experiment instance
exp = Experiment(
    prefix="training-run",
    project="my-project",
    
)

@exp.run
def train_model(experiment):
    """Training function with automatic experiment management."""

    # Set parameters
    experiment.params.set(
        lr=0.001,
        epochs=50,
        model="resnet50"
    )

    # Training loop
    for epoch in range(50):
        loss = 1.0 / (epoch + 1)  # Simulated loss
        experiment.metrics("train").log(loss=loss, epoch=epoch)
        experiment.log(f"Epoch {epoch}: loss={loss:.4f}")

    # Save model
    experiment.files("models").save_json(
        {"final_loss": loss},
        to="results.json"
    )

    return {"final_loss": loss}

# Run training (experiment automatically managed)
result = train_model()
print(f"Training completed: {result}")
```

### Remote Mode with Team Collaboration

```python
from ml_dash import Experiment

# Connect to shared ML-Dash server
with Experiment(
    prefix="team-experiment",
    project="shared-project",
    remote="http://ml-dash-server:3000",
    user_name="alice",
    description="Collaborative experiment",
    tags=["team", "production"]
).run as exp:

    # All data automatically synced to remote server
    exp.params.set(
        researcher="Alice",
        experiment_type="baseline",
        model="bert-base"
    )

    exp.log("Running on remote server", level="info")

    # Metrics stored in MongoDB
    exp.metrics("train").log(loss=0.5, step=100)

    # Files stored in S3
    exp.files("models").save_json(
        {"config": "bert-base"},
        to="model_config.json"
    )

    exp.log("Data synced to team server", level="info")
```

---

## Error Handling

Experiments handle errors gracefully:

```python
try:
    with Experiment(
        prefix="my-experiment",
        project="test",
        
    ).run as exp:
        exp.log("Starting work")
        exp.params.set(test_param="value")

        # Simulate error
        raise ValueError("Something went wrong")

except ValueError as e:
    print(f"Error: {e}")
    # Experiment is automatically marked as FAILED
    # All data is still saved
```

The experiment status will be set to `FAILED` automatically, and all logs, parameters, and metrics are preserved for debugging.

---

## Best Practices

1. **Use Context Managers**: Prefer `with exp.run as exp:` for automatic lifecycle management
2. **Descriptive Names**: Use clear, descriptive experiment names and project names
3. **Add Tags**: Use tags for easy filtering and organization
4. **Log Liberally**: Log important events, errors, and milestones
5. **Structured Metadata**: Use metadata for searchable, structured information
6. **Organize Files**: Use logical prefix paths for file organization
7. **Version Control**: Use bindrs and tags to track model/data versions
8. **Remote for Teams**: Use remote mode for team collaboration and data persistence

---

## API Summary

### Quick Reference

```python
from ml_dash import Experiment

# Create experiment
exp = Experiment(prefix="exp", project="proj")

# Lifecycle
with exp.run as exp:                    # Context manager
    exp.run.start()                     # Manual start
    exp.run.complete()                  # Manual complete
    exp.run.fail()                      # Mark as failed
    exp.run.cancel()                    # Mark as cancelled

# Parameters
exp.params.set(lr=0.001, bs=32)         # Set parameters
params = exp.params.get()                # Get flattened
params = exp.params.get(flatten=False)   # Get nested

# Logging
exp.log("message")                       # Simple log
exp.log("message", level="info")         # With level
exp.log("msg", epoch=1, loss=0.5)        # With metadata
exp.log().info("message")                # Fluent style

# Metrics
exp.metrics("train").log(loss=0.5, step=1)          # Log with prefix
exp.metrics.log(epoch=1, train=dict(loss=0.5))      # Log with groups
exp.metrics("train").buffer(loss=0.5)               # Buffer for summary
exp.metrics.buffer.log_summary("mean", "std")       # Log summaries
exp.metrics.flush()                                 # Flush pending
data = exp.metrics("train").read(0, 100)            # Read data
stats = exp.metrics("train").stats()                # Get stats

# Files
exp.files("models").save("model.pt")
exp.files("configs").save_json(config, to="config.json")
exp.files("models").save_torch(model, to="model.pt")
files = exp.files("models").list()
path = exp.files(file_id="123").download()
exp.files(file_id="123").delete()

# Auto-Start (dxp) - Quick prototyping
from ml_dash.auto_start import dxp

dxp.log("Already started!")              # Ready to use immediately
dxp.params.set(lr=0.001)                 # Set parameters
dxp.metrics("train").log(loss=0.5)       # Log metrics
dxp.files("models").save("model.pt")     # Upload files
# Auto-completed on Python exit
```
