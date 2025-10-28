# ML-Logger API Documentation

**Version:** 0.1.0

ML-Logger is a minimal, experiment tracking library for machine learning. It provides a simple API to log
parameters, metrics, files, and logs during your ML experiments.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
    - [Experiment](#experiment)
    - [Parameters](#parameters)
    - [Metrics](#metrics)
    - [Logs](#logs)
    - [Files](#files)
5. [Usage Patterns](#usage-patterns)
6. [Examples](#examples)
7. [Remote Backend](#remote-backend)
8. [Best Practices](#best-practices)

---

## Installation

```bash
# Using pip
pip install -i https://test.pypi.org/simple/ ml-logger-beta
```

---

## Quick Start

### Basic Example

```python
from ml_dash import Experiment

# Create an experiment
exp = Experiment(
    namespace="alice",  # Your username or team
    workspace="my-project",  # Project name
    prefix="experiment-1"  # Experiment name
)

# Start tracking
with exp.run():
    # Log parameters
    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )

    # Log metrics
    for epoch in range(100):
        exp.metrics.log(
            step=epoch,
            loss=0.5 - epoch * 0.01,
            accuracy=0.5 + epoch * 0.005
        )

    # Log messages
    exp.info("Training completed!")

    # Save files
    exp.files.save({"final": "results"}, "results.json")
```

This creates a local directory structure:

```
.ml-logger/
└── alice/
    └── my-project/
        └── experiment-1/
            ├── .ml-logger.meta.json
            ├── parameters.jsonl
            ├── metrics.jsonl
            ├── logs.jsonl
            └── files/
                └── results.json
```

---

## Core Concepts

### 1. **Namespace**

Your username or organization (e.g., `"alice"`, `"research-team"`)

### 2. **Workspace**

Project or research area (e.g., `"image-classification"`, `"nlp-experiments"`)

### 3. **Prefix**

Unique experiment name (e.g., `"resnet50-run-001"`)

### 4. **Directory** (Optional)

Hierarchical organization within workspace (e.g., `"models/resnet/cifar10"`)

### 5. **Local or Remote**

Everything is saved locally by default. Remote sync is optional.

### 6. **Append-Only**

All data is written in append-only JSONL format for crash safety.

---

## API Reference

### Experiment

The main class for experiment tracking.

#### Constructor

```python
Experiment(
    namespace: str,  # Required: User/team namespace
workspace: str,  # Required: Project workspace
prefix: str,  # Required: Experiment name
remote: str = None,  # Optional: Remote server URL
local_root: str = ".ml-logger",  # Local storage directory
directory: str = None,  # Optional: Subdirectory path
readme: str = None,  # Optional: Description
experiment_id: str = None,  # Optional: Server experiment ID
)
```

**Example:**

```python
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="resnet-experiment",
    directory="image-classification/cifar10",
    readme="ResNet50 transfer learning on CIFAR-10",
    remote="https://qwqdug4btp.us-east-1.awsapprunner.com"  # Optional remote server
)
```

#### Methods

##### `run(func=None)`

Mark experiment as running. Supports 3 patterns:

**Pattern 1: Direct Call**

```python
exp.run()
# ... your training code ...
exp.complete()
```

**Pattern 2: Context Manager (Recommended)**

```python
with exp.run():
# ... your training code ...
# Automatically calls complete() on success
# Automatically calls fail() on exception
```

**Pattern 3: Decorator**

```python
@exp.run
def train():


# ... your training code ...

train()
```

##### `complete()`

Mark experiment as completed.

```python
exp.complete()
```

##### `fail(error: str)`

Mark experiment as failed with error message.

```python
try:
# training code
except Exception as e:
    exp.fail(str(e))
    raise
```

##### Logging Convenience Methods

```python
exp.info(message: str, ** context)  # Log info message
exp.warning(message: str, ** context)  # Log warning message
exp.error(message: str, ** context)  # Log error message
exp.debug(message: str, ** context)  # Log debug message
```

**Example:**

```python
exp.info("Epoch completed", epoch=5, loss=0.3, accuracy=0.85)
exp.warning("Memory usage high", usage_gb=15.2)
exp.error("Training failed", error="CUDA out of memory")
```

#### Properties

```python
exp.namespace  # str: Namespace
exp.workspace  # str: Workspace
exp.prefix  # str: Experiment prefix
exp.directory  # str | None: Directory path
exp.remote  # str | None: Remote server URL
exp.experiment_id  # str | None: Server experiment ID
exp.run_id  # str | None: Server run ID
```

#### Components

```python
exp.params  # ParameterManager
exp.metrics  # MetricsLogger
exp.files  # FileManager
exp.logs  # LogManager
```

---

### Parameters

Manages experiment parameters (hyperparameters, config, etc.)

#### Methods

##### `set(**kwargs)`

Set parameters (replaces existing).

```python
exp.params.set(
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam",
    model={
        "layers": 50,
        "dropout": 0.2
    }
)
```

##### `extend(**kwargs)`

Extend parameters (deep merge with existing).

```python
# First call
exp.params.set(model={"layers": 50})

# Extend (merges with existing)
exp.params.extend(model={"dropout": 0.2})

# Result: {"model": {"layers": 50, "dropout": 0.2}}
```

##### `update(key: str, value: Any)`

Update a single parameter (supports dot notation).

```python
exp.params.update("model.layers", 100)
exp.params.update("learning_rate", 0.0001)
```

##### `read() -> dict`

Read current parameters.

```python
params = exp.params.read()
print(params["learning_rate"])  # 0.001
```

##### `log(**kwargs)`

Alias for `set()` (for API consistency).

```python
exp.params.log(batch_size=64)
```

---

### Metrics

Logs time-series metrics with optional namespacing.

#### Methods

##### `log(step=None, **metrics)`

Log metrics immediately.

```python
# Simple logging
exp.metrics.log(step=1, loss=0.5, accuracy=0.8)

# Multiple metrics at once
exp.metrics.log(
    step=10,
    train_loss=0.3,
    val_loss=0.4,
    train_acc=0.85,
    val_acc=0.82
)

# Without step (uses timestamp only)
exp.metrics.log(gpu_memory=8.5, cpu_usage=45.2)
```

##### `collect(step=None, **metrics)`

Collect metrics for later aggregation (useful for batch-level logging).

```python
for batch in train_loader:
    loss = train_batch(batch)

    # Collect batch metrics (not logged yet)
    exp.metrics.collect(loss=loss.item(), accuracy=acc.item())

# Aggregate and log after epoch
exp.metrics.flush(_aggregation="mean", step=epoch)
```

##### `flush(_aggregation="mean", step=None, **additional_metrics)`

Flush collected metrics with aggregation.

**Aggregation methods:**

- `"mean"` - Average of collected values (default)
- `"sum"` - Sum of collected values
- `"min"` - Minimum value
- `"max"` - Maximum value
- `"last"` - Last value

```python
# Collect during training
for batch in batches:
    metrics.collect(loss=loss, accuracy=acc)

# Flush with mean aggregation
metrics.flush(_aggregation="mean", step=epoch, learning_rate=lr)

# Flush with max aggregation
metrics.flush(_aggregation="max", step=epoch)
```

##### Namespacing: `__call__(namespace: str)`

Create a namespaced metrics logger.

```python
# Create namespaced loggers
train_metrics = exp.metrics("train")
val_metrics = exp.metrics("val")

# Log to different namespaces
train_metrics.log(step=1, loss=0.5, accuracy=0.8)
val_metrics.log(step=1, loss=0.6, accuracy=0.75)

# Results in metrics named: "train.loss", "train.accuracy", "val.loss", "val.accuracy"
```

##### `read() -> list`

Read all logged metrics.

```python
metrics_data = exp.metrics.read()
for entry in metrics_data:
    print(entry["step"], entry["metrics"])
```

---

### Logs

Structured text logging with levels and context.

#### Methods

##### `log(message: str, level: str = "INFO", **context)`

Log a message with level and context.

```python
exp.logs.log("Training started", level="INFO", epoch=0, lr=0.001)
```

##### Level-Specific Methods

```python
exp.logs.info(message: str, ** context)  # INFO level
exp.logs.warning(message: str, ** context)  # WARNING level
exp.logs.error(message: str, ** context)  # ERROR level
exp.logs.debug(message: str, ** context)  # DEBUG level
```

**Examples:**

```python
# Info log
exp.logs.info("Epoch started", epoch=5, batches=100)

# Warning log
exp.logs.warning("High memory usage", memory_gb=14.5, threshold_gb=16.0)

# Error log
exp.logs.error("Training failed", error="CUDA OOM", batch_size=128)

# Debug log
exp.logs.debug("Gradient norm", grad_norm=2.3, step=1000)
```

##### `read() -> list`

Read all logs.

```python
logs = exp.logs.read()
for log_entry in logs:
    print(f"[{log_entry['level']}] {log_entry['message']}")
    if 'context' in log_entry:
        print(f"  Context: {log_entry['context']}")
```

---

### Files

Manages file storage with auto-format detection.

#### Methods

##### `save(data: Any, filename: str)`

Save data with automatic format detection.

**Supported formats:**

- `.json` - JSON files
- `.pkl`, `.pickle` - Pickle files
- `.pt`, `.pth` - PyTorch tensors/models
- `.npy`, `.npz` - NumPy arrays
- Other extensions - Raw bytes or fallback to pickle

```python
# JSON
exp.files.save({"results": [1, 2, 3]}, "results.json")

# PyTorch model
exp.files.save(model.state_dict(), "model.pt")

# NumPy array
exp.files.save(numpy_array, "embeddings.npy")

# Pickle
exp.files.save(custom_object, "object.pkl")

# Raw bytes
exp.files.save(b"binary data", "data.bin")

# Text
exp.files.save("text content", "notes.txt")
```

##### `save_pkl(data: Any, filename: str)`

Save as pickle (automatically adds .pkl extension).

```python
exp.files.save_pkl(complex_object, "checkpoint")
# Saves as "checkpoint.pkl"
```

##### `load(filename: str) -> Any`

Load data with automatic format detection.

```python
# JSON
results = exp.files.load("results.json")

# PyTorch
state_dict = exp.files.load("model.pt")

# NumPy
array = exp.files.load("embeddings.npy")

# Pickle
obj = exp.files.load("object.pkl")
```

##### `load_torch(filename: str) -> Any`

Load PyTorch checkpoint (adds .pt extension if missing).

```python
checkpoint = exp.files.load_torch("best_model")
# Loads "best_model.pt"
```

##### Namespacing: `__call__(namespace: str)`

Create a namespaced file manager.

```python
# Create namespaced file managers
checkpoints = exp.files("checkpoints")
configs = exp.files("configs")

# Save to different directories
checkpoints.save(model.state_dict(), "epoch_10.pt")
# Saves to: files/checkpoints/epoch_10.pt

configs.save(config, "training.json")
# Saves to: files/configs/training.json
```

##### `exists(filename: str) -> bool`

Check if file exists.

```python
if exp.files.exists("checkpoint.pt"):
    model.load_state_dict(exp.files.load("checkpoint.pt"))
```

##### `list() -> list`

List files in current namespace.

```python
files = exp.files.list()
print(f"Files: {files}")

# With namespace
checkpoint_files = exp.files("checkpoints").list()
```

---

## Usage Patterns

### Pattern 1: Simple Training Loop

```python
from ml_dash import Experiment

exp = Experiment(
    namespace="alice",
    workspace="mnist",
    prefix="simple-cnn"
)

with exp.run():
    # Log hyperparameters
    exp.params.set(lr=0.001, epochs=10, batch_size=32)

    # Training loop
    for epoch in range(10):
        train_loss = train_one_epoch()
        val_loss = validate()

        exp.metrics.log(
            step=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )

    # Save model
    exp.files.save(model.state_dict(), "final_model.pt")
```

### Pattern 2: Batch-Level Metrics with Aggregation

```python
with exp.run():
    exp.params.set(lr=0.001, batch_size=128)

    for epoch in range(100):
        # Collect batch-level metrics
        for batch in train_loader:
            loss, acc = train_step(batch)
            exp.metrics.collect(loss=loss, accuracy=acc)

        # Aggregate and log epoch metrics
        exp.metrics.flush(_aggregation="mean", step=epoch)
```

### Pattern 3: Separate Train/Val Metrics

```python
with exp.run():
    # Create namespaced loggers
    train_metrics = exp.metrics("train")
    val_metrics = exp.metrics("val")

    for epoch in range(100):
        # Training phase
        for batch in train_loader:
            loss, acc = train_step(batch)
            train_metrics.collect(loss=loss, accuracy=acc)
        train_metrics.flush(_aggregation="mean", step=epoch)

        # Validation phase
        val_loss, val_acc = validate()
        val_metrics.log(step=epoch, loss=val_loss, accuracy=val_acc)
```

### Pattern 4: Checkpoint Management

```python
with exp.run():
    checkpoints = exp.files("checkpoints")

    best_val_loss = float('inf')

    for epoch in range(100):
        train_loss = train()
        val_loss = validate()

        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoints.save(model.state_dict(), f"epoch_{epoch}.pt")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss
            }, "best_model.pt")

    # Save final model
    checkpoints.save(model.state_dict(), "final_model.pt")
```

### Pattern 5: Hierarchical Organization

```python
# Organize experiments in a directory hierarchy
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="run-001",
    directory="image-classification/resnet50/cifar10"
)
# Creates: .ml-logger/alice/vision/image-classification/resnet50/cifar10/run-001/
```

---

## Examples

### Example 1: Basic MNIST Training

```python
from ml_dash import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create experiment
exp = Experiment(
    namespace="alice",
    workspace="mnist",
    prefix="basic-cnn-001"
)

# Define model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1600, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Training
with exp.run():
    # Log configuration
    exp.params.set(
        learning_rate=0.001,
        batch_size=64,
        epochs=10,
        optimizer="adam"
    )

    # Setup
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=64, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        # Log epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        exp.metrics.log(
            step=epoch,
            loss=avg_loss,
            accuracy=accuracy
        )

        exp.info(f"Epoch {epoch}", loss=avg_loss, accuracy=accuracy)

    # Save final model
    exp.files.save(model.state_dict(), "model.pt")
    exp.info("Training completed!")
```

### Example 2: Transfer Learning with Checkpointing

```python
from ml_dash import Experiment
from torchvision import models
import torch.nn as nn

exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="resnet-transfer-001",
    directory="transfer-learning/cifar10",
    readme="ResNet50 transfer learning on CIFAR-10"
)

with exp.run():
    # Configuration
    config = {
        "model": "resnet50",
        "pretrained": True,
        "num_classes": 10,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 128,
        "early_stopping_patience": 10
    }

    exp.params.set(**config)

    # Load pretrained model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Create namespaced loggers
    train_metrics = exp.metrics("train")
    val_metrics = exp.metrics("val")
    checkpoints = exp.files("checkpoints")

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        for batch in train_loader:
            loss, acc = train_step(model, batch)
            train_metrics.collect(loss=loss, accuracy=acc)

        train_metrics.flush(_aggregation="mean", step=epoch)

        # Validation phase
        model.eval()
        val_loss, val_acc = validate(model, val_loader)
        val_metrics.log(step=epoch, loss=val_loss, accuracy=val_acc)

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoints.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "config": config
            }, "best_model.pt")

            exp.info("New best model!", epoch=epoch, val_acc=val_acc)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            exp.info("Early stopping", epoch=epoch)
            break

        # Regular checkpoint
        if epoch % 10 == 0:
            checkpoints.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    # Save final summary
    exp.files.save({
        "best_val_accuracy": best_val_acc,
        "total_epochs": epoch + 1,
        "config": config
    }, "summary.json")

    exp.info("Training completed!", best_val_acc=best_val_acc)
```

### Example 3: Hyperparameter Sweep

```python
from ml_dash import Experiment

# Define hyperparameter grid
learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        # Create unique experiment for each combination
        exp = Experiment(
            namespace="alice",
            workspace="hp-sweep",
            prefix=f"lr{lr}_bs{bs}",
            directory="mnist/grid-search"
        )

        with exp.run():
            # Log this combination
            exp.params.set(
                learning_rate=lr,
                batch_size=bs,
                model="simple-cnn"
            )

            # Train with these hyperparameters
            final_acc = train_model(lr, bs)

            # Log final result
            exp.metrics.log(step=0, final_accuracy=final_acc)
            exp.info("Sweep run completed", lr=lr, bs=bs, acc=final_acc)

print("Hyperparameter sweep completed!")
```

### Example 4: Multi-Stage Training

```python
from ml_dash import Experiment

exp = Experiment(
    namespace="alice",
    workspace="nlp",
    prefix="bert-finetuning-001",
    directory="transformers/bert/squad"
)

with exp.run():
    # Stage 1: Warmup
    exp.params.set(stage="warmup", lr=0.00001, epochs=5)
    exp.info("Starting warmup phase")

    warmup_metrics = exp.metrics("warmup")
    for epoch in range(5):
        loss = train_epoch(lr=0.00001)
        warmup_metrics.log(step=epoch, loss=loss)

    # Stage 2: Main training
    exp.params.extend(stage="main", lr=0.0001, epochs=20)
    exp.info("Starting main training phase")

    train_metrics = exp.metrics("train")
    val_metrics = exp.metrics("val")

    for epoch in range(20):
        train_loss = train_epoch(lr=0.0001)
        val_loss = validate()

        train_metrics.log(step=epoch, loss=train_loss)
        val_metrics.log(step=epoch, loss=val_loss)

    # Stage 3: Fine-tuning
    exp.params.extend(stage="finetune", lr=0.00001, epochs=10)
    exp.info("Starting fine-tuning phase")

    finetune_metrics = exp.metrics("finetune")
    for epoch in range(10):
        loss = train_epoch(lr=0.00001)
        finetune_metrics.log(step=epoch, loss=loss)

    exp.info("Multi-stage training completed!")
```

---

## Remote Backend

ML-Logger supports syncing to a remote server for team collaboration.

### Setup Remote Backend

```python
exp = Experiment(
    namespace="alice",
    workspace="shared-project",
    prefix="experiment-001",
    remote="http://qwqdug4btp.us-east-1.awsapprunner.com",  # Remote server URL
    readme="Shared experiment for team"
)
```

### How It Works

1. **Local or Remote **: Data can be saved locally or remotely
2. **Automatic Sync**: When `remote` is specified, data syncs to server
3. **Experiment Creation**: Server creates an experiment record
4. **Run Tracking**: Server tracks run status (RUNNING, COMPLETED, FAILED)
5. **GraphQL API**: Query experiments via GraphQL at `http://qwqdug4btp.us-east-1.awsapprunner.com/graphql`

### Environment Variables

Configure remote backend via environment:

```bash
export ML_LOGGER_REMOTE="http://qwqdug4btp.us-east-1.awsapprunner.com"
export ML_LOGGER_NAMESPACE="alice"
export ML_LOGGER_WORKSPACE="production"
```

```python
# Uses environment variables if not specified
exp = Experiment(prefix="my-experiment")
```

### Server Requirements

To use remote backend, you need the dash-server running:

```bash
cd ml-dash/ml-dash-server
pnpm install
pnpm dev
```

Server will be available at `http://qwqdug4btp.us-east-1.awsapprunner.com`

---

## Best Practices

### 1. **Use Context Manager**

Always use `with exp.run():` for automatic cleanup:

```python
# Good
with exp.run():
    train()

# Avoid
exp.run()
train()
exp.complete()  # Easy to forget!
```

### 2. **Namespace Metrics and Files**

Organize metrics and files with namespaces:

```python
train_metrics = exp.metrics("train")
val_metrics = exp.metrics("val")
test_metrics = exp.metrics("test")

checkpoints = exp.files("checkpoints")
configs = exp.files("configs")
visualizations = exp.files("plots")
```

### 3. **Use collect() + flush() for Batch Metrics**

For fine-grained batch logging with epoch aggregation:

```python
for epoch in range(epochs):
    for batch in batches:
        loss = train_batch(batch)
        exp.metrics.collect(loss=loss)

    # Log aggregated metrics once per epoch
    exp.metrics.flush(_aggregation="mean", step=epoch)
```

### 4. **Log Configuration Early**

Log all hyperparameters at the start:

```python
with exp.run():
    exp.params.set(
        model="resnet50",
        learning_rate=0.001,
        batch_size=128,
        epochs=100,
        optimizer="adam",
        dataset="cifar10"
    )
    # ... training ...
```

### 5. **Use Hierarchical Organization**

Organize experiments with directories:

```python
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="run-001",
    directory="models/resnet/cifar10"  # Hierarchical organization
)
```

### 6. **Add Context to Logs**

Make logs searchable with context:

```python
exp.info("Epoch completed",
         epoch=5,
         train_loss=0.3,
         val_loss=0.35,
         learning_rate=0.001)

exp.warning("High memory usage",
            memory_gb=14.5,
            available_gb=16.0,
            batch_size=128)
```

### 7. **Save Comprehensive Checkpoints**

Include all state needed for resumption:

```python
checkpoints.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "config": config
}, "checkpoint.pt")
```

### 8. **Version Control Integration**

Log git information:

```python
import subprocess

git_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode().strip()

exp.params.set(
    git_commit=git_hash,
    git_branch="main"
)
```

### 9. **Error Handling**

The context manager handles errors automatically, but you can add custom handling:

```python
with exp.run():
    try:
        train()
    except RuntimeError as e:
        exp.error("Training error", error=str(e), device="cuda:0")
        raise
```

---

## File Format Details

### metrics.jsonl

```json
{
  "timestamp": 1234567890.123,
  "step": 0,
  "metrics": {
    "loss": 0.5,
    "accuracy": 0.8
  }
}
{
  "timestamp": 1234567891.456,
  "step": 1,
  "metrics": {
    "loss": 0.4,
    "accuracy": 0.85
  }
}
```

### parameters.jsonl

```json
{
  "timestamp": 1234567890.123,
  "operation": "set",
  "data": {
    "lr": 0.001,
    "batch_size": 32
  }
}
{
  "timestamp": 1234567892.456,
  "operation": "update",
  "key": "lr",
  "value": 0.0001
}
```

### logs.jsonl

```json
{
  "timestamp": 1234567890.123,
  "level": "INFO",
  "message": "Training started",
  "context": {
    "epoch": 0
  }
}
{
  "timestamp": 1234567891.456,
  "level": "WARNING",
  "message": "High memory",
  "context": {
    "memory_gb": 14.5
  }
}
```

### .ml-logger.meta.json

```json
{
  "namespace": "alice",
  "workspace": "vision",
  "prefix": "experiment-1",
  "status": "completed",
  "started_at": 1234567890.123,
  "completed_at": 1234567900.456,
  "readme": "ResNet experiment",
  "experiment_id": "exp_123",
  "hostname": "gpu-server-01"
}
```

---

## Troubleshooting

### Issue: Remote server connection failed

```python
# Warning: Failed to initialize experiment on remote server
# Solution: Check if ml-dash-server is running
cd
ml - dash / dash - server & & pnpm
dev
```

### Issue: File not found when loading

```python
# Check if file exists first
if exp.files.exists("model.pt"):
    model.load_state_dict(exp.files.load("model.pt"))
else:
    print("Checkpoint not found")
```

### Issue: Metrics not aggregating correctly

```python
# Make sure to call flush() after collect()
for batch in batches:
    metrics.collect(loss=loss)

metrics.flush(_aggregation="mean", step=epoch)  # Don't forget this!
```

---

## API Summary

| Component      | Key Methods                                 | Purpose                     |
|----------------|---------------------------------------------|-----------------------------|
| **Experiment** | `run()`, `complete()`, `fail()`, `info()`   | Manage experiment lifecycle |
| **Parameters** | `set()`, `extend()`, `update()`, `read()`   | Store configuration         |
| **Metrics**    | `log()`, `collect()`, `flush()`, `read()`   | Track time-series metrics   |
| **Logs**       | `info()`, `warning()`, `error()`, `debug()` | Structured logging          |
| **Files**      | `save()`, `load()`, `exists()`, `list()`    | File management             |

---

## Additional Resources

- **GitHub**: https://github.com/vuer-ai/vuer-dashboard
- **Examples**: See `ml-logger/examples/` directory
- **Tests**: See `ml-logger/tests/` for usage examples
- **Dashboard**: http://qwqdug4btp.us-east-1.awsapprunner.com (when dash-server is running)

---

## Contributing & Development

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vuer-ai/vuer-dashboard.git
   cd vuer-dashboard/ml-logger
   ```

2. **Install with development dependencies**:
   ```bash
   # Install all dev dependencies (includes testing, docs, and torch)
   uv sync --extra dev
   ```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ml_dash --cov-report=html

# Run specific test file
uv run pytest tests/test_backends.py
```

### Building Documentation

```bash
# Build HTML documentation
cd docs && make html

# Serve docs with live reload (auto-refreshes on file changes)
cd docs && make serve

# Clean build artifacts
cd docs && make clean
```

The built documentation will be in `docs/_build/html/`. The `make serve` command starts a local server at `http://localhost:8000` with automatic rebuilding on file changes.

### Linting and Code Checks

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Project Structure

```
ml-logger/
├── src/ml_logger_beta/          # Main package source
│   ├── __init__.py              # Package exports
│   ├── run.py                   # Experiment class
│   ├── ml_logger.py             # ML_Logger class
│   ├── job_logger.py            # JobLogger class
│   ├── backends/                # Storage backends
│   │   ├── base.py              # Base backend interface
│   │   ├── local_backend.py     # Local filesystem backend
│   │   └── dash_backend.py      # Remote server backend
│   └── components/              # Component managers
│       ├── parameters.py        # Parameter management
│       ├── metrics.py           # Metrics logging
│       ├── logs.py              # Structured logging
│       └── files.py             # File management
├── tests/                       # Test suite
├── docs/                        # Sphinx documentation
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

### Dependency Structure

The project uses a simplified dependency structure:

- **`dependencies`**: Core runtime dependencies (always installed)
    - `msgpack`, `numpy`, `requests`
- **`dev`**: All development dependencies
    - Linting and formatting: `ruff`
    - Testing: `pytest`, `pytest-cov`, `pytest-asyncio`
    - Documentation: `sphinx`, `furo`, `myst-parser`, `sphinx-copybutton`, `sphinx-autobuild`
    - Optional features: `torch` (for saving/loading .pt/.pth files)

### Making Changes

1. Create a new branch for your changes
2. Make your modifications
3. Run tests to ensure everything works: `uv run pytest`
4. Run linting: `uv run ruff check .`
5. Format code: `uv run ruff format .`
6. Update documentation if needed
7. Submit a pull request

### Building and Publishing

```bash
# Build the package
uv build

# Publish to PyPI (requires credentials)
uv publish
```

### Tips for Contributors

- Follow the existing code style (enforced by ruff)
- Add tests for new features
- Update documentation for API changes
- Use type hints where appropriate
- Keep functions focused and modular
- Write descriptive commit messages

---

**Happy Experimenting!** 🚀
