# ML-Dash Tutorial

A complete guide to experiment tracking with ML-Dash - from basics to advanced features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Organizing Experiments](#organizing-experiments)
4. [Logging Data](#logging-data)
5. [Remote Backend](#remote-backend)
6. [Real-World Examples](#real-world-examples)
7. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
cd ml-dash
uv sync

# Using pip
pip install -e .
```

### Your First Experiment (60 seconds)

```python
from ml_dash import Experiment

# Create an experiment
exp = Experiment(
    namespace="your-username",  # Your username or team
    workspace="my-project",  # Project name
    prefix="first-experiment"  # Experiment name
)

# Start the experiment
with exp.run():
    # Log hyperparameters
    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Log training metrics
    for epoch in range(10):
        exp.metrics.log(
            step=epoch,
            loss=1.0 / (epoch + 1),  # Simulated loss
            accuracy=epoch * 0.1  # Simulated accuracy
        )

    # Save results
    results = {"final_accuracy": 0.95}
    exp.files.save(results, "results.json")

    # Log completion
    exp.info("Training completed!")

print(f"✓ Experiment saved to: {exp.local_path}")
# Output: ✓ Experiment saved to: your-username/my-project/first-experiment
```

**That's it!** Your experiment data is now saved locally in `.ml-dash/`

---

## Core Concepts

### The Experiment Object

An `Experiment` tracks everything about a training run:

```python
from ml_dash import Experiment

exp = Experiment(
    namespace="alice",  # User or team namespace
    workspace="vision-project",  # Project/workspace name
    prefix="resnet50-baseline",  # Experiment identifier
    readme="Training ResNet50 on CIFAR-10",  # Optional description
)

# Four main components:
exp.params  # Hyperparameters and configuration
exp.metrics  # Time-series data (loss, accuracy, etc.)
exp.files  # Model checkpoints and artifacts
exp.logs  # Text logs and events
```

### Storage Structure

ML-Dash stores everything locally by default:

```
.ml-dash/
└── alice/                          # namespace
    └── vision-project/             # workspace
        └── resnet50-baseline/      # prefix (experiment name)
            ├── parameters.jsonl    # Hyperparameters
            ├── metrics.jsonl       # Training metrics
            ├── logs.jsonl          # Text logs
            ├── files/              # Saved files
            │   ├── model.pt
            │   └── config.json
            └── .ml-dash.meta.json  # Metadata
```

### Three Ways to Run Experiments

#### 1. Context Manager (Recommended)

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp1")

with exp.run():
    # Your training code here
    exp.params.set(lr=0.001)
    exp.metrics.log(step=0, loss=0.5)

# Automatically marked as completed (or failed if exception)
```

#### 2. Decorator

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp2")

@exp.run
def train(config):
    exp.params.set(**config)
    for epoch in range(config["epochs"]):
        loss = train_epoch()
        exp.metrics.log(step=epoch, loss=loss)
    return loss

final_loss = train({"epochs": 10, "lr": 0.001})
```

#### 3. Manual Control

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp3")

exp.run()  # Mark as started
# Your training code
exp.complete()  # Mark as completed
```

---

## Organizing Experiments

### Directory Organization (NEW!)

Organize experiments hierarchically with the `directory` parameter:

```python
from ml_dash import Experiment

# Single directory level
exp = Experiment(
    namespace="research-team",
    workspace="computer-vision",
    prefix="resnet50-run1",
    directory="image-classification"
)
# Stored at: research-team/computer-vision/image-classification/resnet50-run1/

# Nested directories (recommended for large projects)
exp = Experiment(
    namespace="research-team",
    workspace="nlp",
    prefix="bert-base",
    directory="transformers/bert/squad"
)
# Stored at: research-team/nlp/transformers/bert/squad/bert-base/
```

### Organization Strategies

#### By Algorithm

```python
# Experiments organized by algorithm type
exp1 = Experiment(
    namespace="team",
    workspace="vision",
    prefix="run-001",
    directory="algorithms/supervised/resnet"
)

exp2 = Experiment(
    namespace="team",
    workspace="vision",
    prefix="run-001",
    directory="algorithms/supervised/vit"
)

exp3 = Experiment(
    namespace="team",
    workspace="vision",
    prefix="run-001",
    directory="algorithms/self-supervised/simclr"
)
```

#### By Dataset

```python
# Experiments organized by dataset
exp = Experiment(
    namespace="alice",
    workspace="image-classification",
    prefix="resnet50",
    directory="datasets/imagenet/baseline"
)

exp = Experiment(
    namespace="alice",
    workspace="image-classification",
    prefix="resnet50",
    directory="datasets/cifar10/baseline"
)
```

#### By Project Phase

```python
# Experiments organized by development phase
exp = Experiment(
    namespace="team",
    workspace="production-model",
    prefix="version-1.0",
    directory="phases/development"
)

exp = Experiment(
    namespace="team",
    workspace="production-model",
    prefix="version-1.0",
    directory="phases/staging"
)

exp = Experiment(
    namespace="team",
    workspace="production-model",
    prefix="version-1.0",
    directory="phases/production"
)
```

#### Mixed Organization

```python
# Combine multiple organization strategies
exp = Experiment(
    namespace="research-lab",
    workspace="autonomous-driving",
    prefix="run-042",
    directory="2024/Q4/object-detection/yolo/v8"
)
# Organized by: year/quarter/task/algorithm/version
```

### Hierarchical Prefixes

Use forward slashes in prefixes for additional organization:

```python
# Old style - flat
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="resnet50-cifar10-run001"
)

# New style - hierarchical with directory
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="run-001",
    directory="models/resnet50/cifar10"
)
# Cleaner and more organized!
```

### Multiple Experiments in Same Directory

```python
# Run multiple experiments in the same directory
for run_id in range(5):
    exp = Experiment(
        namespace="team",
        workspace="hyperparameter-search",
        prefix=f"run-{run_id:03d}",
        directory="models/resnet50/cifar10",  # Shared directory
    )

    with exp.run():
        # Each run has different hyperparameters
        lr = 0.001 * (2 ** run_id)
        exp.params.set(learning_rate=lr, run_id=run_id)

        # Train and log metrics
        train_model(lr)

# Result:
# team/hyperparameter-search/models/resnet50/cifar10/
#   ├── run-000/
#   ├── run-001/
#   ├── run-002/
#   ├── run-003/
#   └── run-004/
```

---

## Logging Data

### Parameters (Hyperparameters & Config)

Parameters are your experiment configuration - they don't change during training.

#### Basic Parameters

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp")

# Set all parameters at once
exp.params.set(
    learning_rate=0.001,
    batch_size=128,
    epochs=100,
    optimizer="adam",
    weight_decay=0.0001
)

# Read parameters back
params = exp.params.read()
print(params["learning_rate"])  # 0.001
```

#### Nested Parameters (Recommended)

Organize parameters hierarchically:

```python
exp.params.set(
    # Model configuration
    model=dict(
        name="resnet50",
        num_layers=50,
        hidden_dim=2048,
        dropout=0.5,
        pretrained=True
    ),

    # Training configuration
    training=dict(
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        lr_schedule="cosine",
        warmup_epochs=5
    ),

    # Data configuration
    data=dict(
        dataset="imagenet",
        image_size=224,
        augmentation=True,
        num_workers=8
    ),

    # System configuration
    system=dict(
        gpu_ids=[0, 1, 2, 3],
        mixed_precision=True,
        distributed=True
    )
)

# Access nested parameters
params = exp.params.read()
print(params["model"]["name"])           # "resnet50"
print(params["training"]["learning_rate"])  # 0.001
```

#### Updating Parameters

```python
# Extend (adds new parameters, keeps existing)
exp.params.extend(
    training=dict(
        early_stopping=True,  # Adds to training dict
        patience=10
    )
)

# Update single parameter (dot notation supported)
exp.params.update("learning_rate", 0.0005)
exp.params.update("model.dropout", 0.3)
exp.params.update("training.batch_size", 256)
```

### Metrics (Time-Series Data)

Metrics are values that change over time during training.

#### Basic Metrics

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp")

with exp.run():
    # Log metrics at each step
    for epoch in range(100):
        train_loss = train_one_epoch()  # Your training code
        val_loss, val_acc = validate()  # Your validation code

        exp.metrics.log(
            step=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=get_current_lr()
        )
```

#### Collecting and Aggregating Batch Metrics

For mini-batch training, collect metrics and aggregate at epoch level:

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp")

with exp.run():
    for epoch in range(10):
        # Training phase - collect batch metrics
        for batch_idx, batch in enumerate(train_loader):
            loss, acc = train_step(batch)

            # Collect (don't log yet)
            exp.metrics.collect(
                batch_loss=loss,
                batch_accuracy=acc
            )

        # Aggregate and log at epoch level
        exp.metrics.flush(
            _aggregation="mean",  # Options: mean, sum, min, max, last
            step=epoch,

            # Add validation metrics (not aggregated)
            val_loss=validate(),
            val_accuracy=test()
        )

print("✓ Logged aggregated metrics for 10 epochs")
```

#### Namespaced Metrics (Recommended for Complex Training)

Use namespaces to organize metrics by phase:

```python
exp = Experiment(
    namespace="alice",
    workspace="project",
    prefix="exp",
    directory="experiments/phase1"
)

with exp.run():
    # Create namespaced metric loggers
    train_metrics = exp.metrics("train")
    val_metrics = exp.metrics("val")
    test_metrics = exp.metrics("test")

    for epoch in range(10):
        # Log training metrics
        train_metrics.log(
            step=epoch,
            loss=0.5,
            accuracy=0.85,
            lr=0.001
        )

        # Log validation metrics
        val_metrics.log(
            step=epoch,
            loss=0.3,
            accuracy=0.90
        )

    # Final test metrics
    test_metrics.log(
        step=0,
        loss=0.25,
        accuracy=0.92
    )

# Metrics are stored with prefixes:
# - train.loss, train.accuracy, train.lr
# - val.loss, val.accuracy
# - test.loss, test.accuracy
```

#### Multi-Level Namespaces

```python
# Create nested namespaces for complex architectures
model_metrics = exp.metrics("model")
encoder_metrics = model_metrics("encoder")
decoder_metrics = model_metrics("decoder")

encoder_metrics.log(step=0, loss=0.5)  # Stored as "model.encoder.loss"
decoder_metrics.log(step=0, loss=0.3)  # Stored as "model.decoder.loss"
```

### Files (Models, Checkpoints, Artifacts)

Save and load models, checkpoints, and any other files.

#### Automatic Format Detection

ML-Dash auto-detects format based on file extension:

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp")

with exp.run():
    # JSON files
    config = {"model": "resnet50", "dataset": "imagenet"}
    exp.files.save(config, "config.json")

    # Pickle files
    data = {"results": [1, 2, 3], "metadata": {"v": 1}}
    exp.files.save(data, "data.pkl")

    # Text files
    exp.files.save("Training completed successfully!", "summary.txt")

    # Binary files
    binary_data = b"\x00\x01\x02\x03"
    exp.files.save(binary_data, "data.bin")

# Load files back
config = exp.files.load("config.json")
data = exp.files.load("data.pkl")
```

#### PyTorch Models

```python
import torch
import torch.nn as nn

# Create a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Save model state
exp.files.save(model.state_dict(), "model.pt")

# Save full checkpoint
checkpoint = {
    "epoch": 50,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": 0.25,
}
exp.files.save(checkpoint, "checkpoint_epoch_50.pt")

# Load model state
loaded_state = exp.files.load("model.pt")
model.load_state_dict(loaded_state)
```

#### NumPy Arrays

```python
import numpy as np

# Save arrays
embeddings = np.random.randn(10000, 128)
exp.files.save(embeddings, "embeddings.npy")

# Save multiple arrays
exp.files.save(
    {"embeddings": embeddings, "labels": labels},
    "data.npz"
)

# Load arrays
loaded = exp.files.load("embeddings.npy")
```

#### Organized File Storage with Namespaces

```python
# Create namespaced file managers
checkpoints = exp.files("checkpoints")
visualizations = exp.files("visualizations")
configs = exp.files("configs")

# Save to organized directories
checkpoints.save(model.state_dict(), "epoch_10.pt")
# Saved to: files/checkpoints/epoch_10.pt

visualizations.save(plot_data, "loss_curve.json")
# Saved to: files/visualizations/loss_curve.json

configs.save(config, "final_config.json")
# Saved to: files/configs/final_config.json
```

### Logs (Events and Debugging)

Log text messages with different severity levels.

#### Basic Logging

```python
exp = Experiment(namespace="alice", workspace="project", prefix="exp")

with exp.run():
    # Different log levels
    exp.info("Training started")
    exp.debug("Loading batch 100", batch_size=32)
    exp.warning("GPU memory usage: 15.2 GB")
    exp.error("Failed to load checkpoint", error="File not found")

# Shortcut: Use logs component directly
exp.logs.info("Training started")
exp.logs.warning("Low memory")
exp.logs.error("Critical error", details="OOM")
```

#### Structured Logging with Context

Add metadata to logs for better debugging:

```python
with exp.run():
    for epoch in range(100):
        exp.info(
            "Epoch completed",
            epoch=epoch,
            train_loss=0.5,
            val_loss=0.3,
            time_seconds=125.3,
            gpu_memory_gb=14.2
        )

        if epoch % 10 == 0:
            exp.info(
                "Checkpoint saved",
                epoch=epoch,
                path=f"checkpoint_{epoch}.pt",
                model_size_mb=450
            )

    exp.info(
        "Training completed",
        total_epochs=100,
        final_loss=0.05,
        total_time_hours=8.5,
        best_val_accuracy=0.95
    )
```

#### Log Levels

```python
# DEBUG: Detailed information for diagnosing problems
exp.debug("Processing batch", batch_idx=100, batch_size=32)

# INFO: General informational messages
exp.info("Epoch completed", epoch=10, loss=0.5)

# WARNING: Warning messages for potentially harmful situations
exp.warning("High GPU memory usage", usage_gb=15.2)

# ERROR: Error messages for serious problems
exp.error("Failed to save checkpoint", error="Disk full")
```

---

## Remote Backend

Send all experiment data to a centralized server for team collaboration and advanced features.

### Why Use Remote Backend?

**Local Backend (Default):**
- ✓ Works out of the box
- ✓ No setup required
- ✓ Fast (no network)
- ✗ Local files only
- ✗ No collaboration
- ✗ No web dashboard

**Remote Backend:**
- ✓ **All logs stored in MongoDB**
- ✓ **All files uploaded to S3**
- ✓ **Team collaboration**
- ✓ **Web dashboard with GraphQL API**
- ✓ **Real-time synchronization**
- ✗ Requires dash-server setup

### Quick Setup

```bash
# 1. Start MongoDB
cd docker
docker-compose up -d

# 2. Configure ml-dash-server
cd ml-dash/ml-dash-server
# Add your AWS credentials to .env file:
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your-key
# AWS_SECRET_ACCESS_KEY=your-secret
# S3_BUCKET_NAME=your-bucket

# 3. Start ml-dash-server
pnpm dev
# Server runs at http://localhost:4000
```

### Using Remote Backend

Simply add `remote` parameter - everything else stays the same!

```python
from ml_dash import Experiment

# Local backend
exp_local = Experiment(
    namespace="alice",
    workspace="project",
    prefix="local-exp"
)

# Remote backend - just add remote parameter!
exp_remote = Experiment(
    namespace="alice",
    workspace="project",
    prefix="remote-exp",
    remote="http://localhost:4000"  # 👈 This is the only difference!
)

# Everything else works exactly the same
with exp_remote.run():
    exp_remote.params.set(learning_rate=0.001)
    exp_remote.metrics.log(step=0, loss=0.5)
    exp_remote.files.save(model_state, "model.pt")  # Uploaded to S3!
    exp_remote.info("Training completed")

print(f"✓ Experiment ID: {exp_remote.experiment_id}")
print(f"✓ Run ID: {exp_remote.run_id}")
print(f"✓ View at: http://localhost:4000/graphql")
```

### Remote Backend with Directory Organization

Directories work seamlessly with remote backend:

```python
# Remote backend automatically creates directory hierarchy in database
exp = Experiment(
    namespace="research-team",
    workspace="nlp",
    prefix="bert-base",
    directory="transformers/bert/squad",  # Creates 3-level hierarchy
    remote="http://localhost:4000",
    readme="BERT fine-tuning on SQuAD dataset",
)

with exp.run():
    exp.params.set(
        model="bert-base-uncased",
        learning_rate=3e-5,
        batch_size=32
    )

    for epoch in range(5):
        exp.metrics.log(step=epoch, loss=0.5 - epoch * 0.1)

    exp.files.save(model.state_dict(), "model.pt")  # Uploaded to S3

# On the server:
# - Creates Directory: "transformers" (parent=null)
# - Creates Directory: "transformers/bert" (parent=transformers)
# - Creates Directory: "transformers/bert/squad" (parent=bert)
# - Links experiment to the "squad" directory
```

### Environment Variables

Configure once, use everywhere:

```bash
# Set environment variables
export ML_DASH_NAMESPACE="alice"
export ML_DASH_WORKSPACE="my-project"
export ML_DASH_PREFIX="exp-001"
export ML_DASH_REMOTE="http://localhost:4000"
```

```python
from ml_dash.autolog import experiment

# Automatically configured from environment!
experiment.params.set(learning_rate=0.001)
experiment.metrics.log(step=0, loss=0.5)
```

### Production Configuration

```python
import os

# Use environment variables for production
exp = Experiment(
    namespace=os.environ.get("ML_TEAM_NAME", "default-team"),
    workspace=os.environ["PROJECT_NAME"],
    prefix=f"run-{os.environ['RUN_ID']}",
    directory=f"{os.environ['MODEL_TYPE']}/{os.environ['DATASET']}",
    remote=os.environ.get("DASH_SERVER_URL"),
)
```

### Error Handling and Fallback

Remote backend automatically falls back to local storage if server is unavailable:

```python
exp = Experiment(
    namespace="alice",
    workspace="project",
    prefix="exp",
    remote="http://localhost:4000"
)

# If server is down, you'll see:
# Warning: Failed to initialize experiment on remote server: Connection refused
# Falls back to local backend automatically

# Your code continues to work!
with exp.run():
    exp.metrics.log(step=0, loss=0.5)
    # Data saved locally instead of remote server
```

---

## Real-World Examples

### Example 1: Image Classification with ResNet

Complete image classification training with all best practices:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from ml_dash import Experiment

# Configuration
config = {
    "model": {
        "name": "resnet50",
        "pretrained": True,
        "num_classes": 10
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "lr_schedule": "cosine",
        "early_stopping": True,
        "patience": 10
    },
    "data": {
        "dataset": "cifar10",
        "image_size": 224,
        "augmentation": True,
        "num_workers": 4
    }
}

# Create experiment
exp = Experiment(
    namespace="alice",
    workspace="vision",
    prefix="run-001",
    directory="image-classification/resnet50/cifar10",
    remote="http://localhost:4000",  # Use remote backend for team sharing
    readme="ResNet50 on CIFAR-10 with transfer learning",
)

# Training
with exp.run():
    # Log configuration
    exp.params.set(**config)
    exp.info("Training started", config=config)

    # Setup data
    transform_train = transforms.Compose([
        transforms.Resize(config["data"]["image_size"]),
        transforms.RandomCrop(config["data"]["image_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )

    val_loader = DataLoader(...)  # Similar setup for validation

    # Create model
    model = models.resnet50(pretrained=config["model"]["pretrained"])
    model.fc = nn.Linear(model.fc.in_features, config["model"]["num_classes"])
    model = model.cuda()

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"]
    )

    # Create metric loggers
    train_metrics = exp.metrics("train")
    val_metrics = exp.metrics("val")

    # Create file manager for checkpoints
    checkpoints = exp.files("checkpoints")

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            # Collect batch metrics
            acc = (output.argmax(1) == target).float().mean()
            train_metrics.collect(
                loss=loss.item(),
                accuracy=acc.item()
            )

        # Log aggregated training metrics
        train_metrics.flush(
            _aggregation="mean",
            step=epoch,
            learning_rate=scheduler.get_last_lr()[0]
        )

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss += nn.CrossEntropyLoss()(output, target).item()
                val_correct += (output.argmax(1) == target).sum().item()
                val_total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        val_metrics.log(step=epoch, loss=val_loss, accuracy=val_acc)

        # Log epoch summary
        exp.info(
            "Epoch completed",
            epoch=epoch,
            train_loss=train_metrics._collect_buffer.get("loss", [0])[-1],
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=scheduler.get_last_lr()[0]
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoints.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config
                },
                "best_model.pt"
            )
            exp.info("New best model!", epoch=epoch, val_accuracy=val_acc)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["training"]["patience"]:
            exp.info("Early stopping triggered", epoch=epoch, patience=patience_counter)
            break

        # Regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoints.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

        # Update learning rate
        scheduler.step()

    # Save final model
    checkpoints.save(model.state_dict(), "final_model.pt")

    # Save training summary
    summary = {
        "best_val_accuracy": best_val_acc,
        "total_epochs": epoch + 1,
        "config": config
    }
    exp.files.save(summary, "training_summary.json")

    exp.info(
        "Training completed!",
        best_val_accuracy=best_val_acc,
        total_epochs=epoch + 1
    )

print(f"✓ Training completed!")
print(f"  Best validation accuracy: {best_val_acc:.4f}")
print(f"  Experiment ID: {exp.experiment_id}")
print(f"  View at: http://localhost:4000/graphql")
```

### Example 2: Hyperparameter Search

Systematic hyperparameter tuning with ML-Dash:

```python
from ml_dash import Experiment
import itertools
import numpy as np

# Define hyperparameter grid
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "dropout": [0.3, 0.5, 0.7],
    "weight_decay": [0.0001, 0.001, 0.01]
}

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Testing {len(combinations)} hyperparameter combinations")

# Store results
results = []

for idx, params in enumerate(combinations):
    # Create descriptive experiment name
    exp_name = (
        f"lr{params['learning_rate']}_"
        f"bs{params['batch_size']}_"
        f"drop{params['dropout']}_"
        f"wd{params['weight_decay']}"
    )

    # Create experiment
    exp = Experiment(
        namespace="alice",
        workspace="hyperparameter-search",
        prefix=exp_name,
        directory="resnet50/cifar10/grid-search",
        remote="http://localhost:4000",
    )

    with exp.run():
        # Log hyperparameters
        exp.params.set(**params)

        exp.info(
            "Starting hyperparameter combination",
            combination=f"{idx + 1}/{len(combinations)}",
            params=params
        )

        # Train model (your training code)
        final_val_acc = train_model_with_params(params)

        # Log final result
        exp.metrics.log(step=0, final_val_accuracy=final_val_acc)

        # Store result
        results.append({
            "params": params,
            "val_accuracy": final_val_acc,
            "experiment": exp_name,
            "experiment_id": exp.experiment_id
        })

        exp.info(
            "Combination completed",
            val_accuracy=final_val_acc
        )

# Find best configuration
results.sort(key=lambda x: x["val_accuracy"], reverse=True)
best = results[0]

print(f"\n✓ Hyperparameter search completed!")
print(f"  Best configuration:")
print(f"    Learning rate: {best['params']['learning_rate']}")
print(f"    Batch size: {best['params']['batch_size']}")
print(f"    Dropout: {best['params']['dropout']}")
print(f"    Weight decay: {best['params']['weight_decay']}")
print(f"    Validation accuracy: {best['val_accuracy']:.4f}")
print(f"    Experiment: {best['experiment']}")

# Save results summary
summary_exp = Experiment(
    namespace="alice",
    workspace="hyperparameter-search",
    prefix="summary",
    directory="resnet50/cifar10/grid-search"
)
summary_exp.files.save(results, "all_results.json")
summary_exp.files.save(best, "best_config.json")
```

### Example 3: Multi-GPU Distributed Training

Using ML-Dash with PyTorch distributed training:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ml_dash import Experiment


def train_worker(rank, world_size, config):
    """Training function for each GPU process"""

    # Setup distributed training
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=world_size,
        rank=rank
    )

    # Only rank 0 creates the experiment
    if rank == 0:
        exp = Experiment(
            namespace="team",
            workspace="distributed-training",
            prefix="run-001",
            directory="models/resnet50/imagenet",
            remote="http://localhost:4000",
        )

        exp_run = exp.run()
        exp_run.__enter__()

        # Log configuration
        exp.params.set(
            **config,
            distributed=True,
            world_size=world_size,
            gpus=list(range(world_size))
        )

        exp.info("Distributed training started", world_size=world_size)

    # Create model and wrap with DDP
    model = create_model(config).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Only rank 0 logs metrics
            if rank == 0 and batch_idx % 100 == 0:
                exp.metrics.collect(loss=loss.item())

        # Synchronize and aggregate metrics
        if rank == 0:
            exp.metrics.flush(_aggregation="mean", step=epoch)
            exp.info("Epoch completed", epoch=epoch, rank=rank)

    # Cleanup
    if rank == 0:
        exp.files.save(model.module.state_dict(), "final_model.pt")
        exp_run.__exit__(None, None, None)

    dist.destroy_process_group()


# Launch distributed training
if __name__ == "__main__":
    config = {
        "model": "resnet50",
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001
    }

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
```

### Example 4: Experiment Comparison

Compare multiple experiments programmatically:

```python
from ml_dash import Experiment
import json
from pathlib import Path


def load_experiment_results(namespace, workspace, directory):
    """Load results from all experiments in a directory"""

    base_path = Path(f".ml-dash/{namespace}/{workspace}/{directory}")

    results = []

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Read parameters
        params_file = exp_dir / "parameters.jsonl"
        if params_file.exists():
            with open(params_file) as f:
                last_line = list(f)[-1]
                params = json.loads(last_line)["data"]

        # Read final metrics
        metrics_file = exp_dir / "metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file) as f:
                lines = list(f)
                last_metrics = json.loads(lines[-1])["metrics"]

        results.append({
            "experiment": exp_dir.name,
            "params": params,
            "final_metrics": last_metrics
        })

    return results


# Compare experiments
results = load_experiment_results(
    namespace="alice",
    workspace="hyperparameter-search",
    directory="resnet50/cifar10/grid-search"
)

# Sort by validation accuracy
results.sort(
    key=lambda x: x["final_metrics"].get("val.accuracy", 0),
    reverse=True
)

# Print top 5
print("Top 5 experiments:")
for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. {result['experiment']}")
    print(f"   Val Accuracy: {result['final_metrics'].get('val.accuracy', 0):.4f}")
    print(f"   Learning Rate: {result['params'].get('learning_rate')}")
    print(f"   Batch Size: {result['params'].get('batch_size')}")
```

---

## Best Practices

### 1. Use Hierarchical Organization

```python
# ✅ Good - Clear hierarchy
exp = Experiment(
    namespace="research-team",
    workspace="computer-vision",
    prefix="run-042",
    directory="2024/Q4/image-classification/resnet50/cifar10"
)

# ❌ Avoid - Flat structure
exp = Experiment(
    namespace="team",
    workspace="project",
    prefix="exp042"
)
```

### 2. Always Use Context Managers

```python
# ✅ Good - Automatic lifecycle management
with exp.run():
    train_model()

# ❌ Avoid - Manual management
exp.run()
train_model()
exp.complete()
```

### 3. Use Nested Parameters

```python
# ✅ Good - Organized configuration
exp.params.set(
    model=dict(name="resnet50", layers=50),
    training=dict(lr=0.001, epochs=100),
    data=dict(dataset="imagenet", batch_size=128)
)

# ❌ Avoid - Flat parameters
exp.params.set(
    model_name="resnet50",
    model_layers=50,
    training_lr=0.001,
    training_epochs=100,
    ...
)
```

### 4. Aggregate Batch Metrics

```python
# ✅ Good - Log aggregated metrics per epoch
for epoch in range(epochs):
    for batch in batches:
        exp.metrics.collect(loss=loss)
    exp.metrics.flush(_aggregation="mean", step=epoch)

# ❌ Avoid - Logging every batch
for epoch in range(epochs):
    for batch_idx, batch in enumerate(batches):
        exp.metrics.log(step=batch_idx, loss=loss)  # Too much data!
```

### 5. Use Namespaces for Complex Training

```python
# ✅ Good - Organized by phase
train_metrics = exp.metrics("train")
val_metrics = exp.metrics("val")
test_metrics = exp.metrics("test")

train_metrics.log(step=0, loss=0.5)
val_metrics.log(step=0, loss=0.3)

# ❌ Avoid - Mixed metrics
exp.metrics.log(step=0, train_loss=0.5, val_loss=0.3, test_loss=0.2)
```

### 6. Save Checkpoints Regularly

```python
# TODO shortcut
checkpoints = exp.files("checkpoints") 

# Save best model
if is_best:
    checkpoints.save(model.state_dict(), "best_model.pt")

# Save periodic checkpoints
if epoch % 10 == 0:
    checkpoints.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

# Always save final model
checkpoints.save(model.state_dict(), "final_model.pt")
```

### 7. Log Important Events

```python
exp.info("Training started", config=config)
exp.info("Epoch completed", epoch=epoch, loss=loss, time=time_taken)
exp.warning("High GPU memory usage", usage_gb=15.2)
exp.error("Training failed", error=str(e))
exp.info("Training completed", total_time=total_time, final_acc=acc)
```

### 8. Use Remote Backend for Production

```python
# Development - local backend
if os.environ.get("ENVIRONMENT") == "development":
    exp = Experiment(
        namespace="alice",
        workspace="dev",
        prefix="test"
        # No remote = local storage
    )

# Production - remote backend
else:
    exp = Experiment(
        namespace="ml-team",
        workspace="production",
        prefix="run-001",
        remote=os.environ["DASH_SERVER_URL"],  # Centralized server
    )
```

### 9. Use Descriptive Names and Documentation

```python
# ✅ Good - Clear and documented
exp = Experiment(
    namespace="research-team",
    workspace="image-classification",
    prefix="run-042",
    directory="models/resnet50/imagenet/2024-Q4",
    readme="""
    ResNet50 training on ImageNet with the following improvements:
    - Mixed precision training (FP16)
    - Label smoothing (α=0.1)
    - Cosine annealing learning rate schedule
    - Gradient clipping (max_norm=1.0)

    Expected to achieve 76.5% top-1 accuracy.
    """
)

# ❌ Avoid - Unclear
exp = Experiment(
    namespace="team",
    workspace="proj",
    prefix="exp42"
)
```

---

## Quick Reference

### Creating Experiments

```python
from ml_dash import Experiment

# Local backend
exp = Experiment(
    namespace="alice",
    workspace="project",
    prefix="experiment-name",
    directory="optional/path",  # Optional directory structure
    readme="Description",  # Optional description
)

# Remote backend (add remote parameter)
exp = Experiment(
    namespace="alice",
    workspace="project",
    prefix="experiment-name",
    directory="optional/path",
    remote="http://localhost:4000",  # 👈 Remote server URL
    readme="Description",
)
```

### Logging Data

```python
# Parameters
exp.params.set(lr=0.001, batch_size=32)
exp.params.extend(model=dict(layers=50))
exp.params.update("lr", 0.0005)

# Metrics
exp.metrics.log(step=0, loss=0.5, accuracy=0.9)
exp.metrics.collect(batch_loss=0.3)
exp.metrics.flush(_aggregation="mean", step=0)

# Files
exp.files.save(data, "file.json")
loaded = exp.files.load("file.json")

# Logs
exp.info("Message", key=value)
exp.warning("Warning", key=value)
exp.error("Error", key=value)
```

### Running Experiments

```python
# Context manager (recommended)
with exp.run():
    train_model()

# Decorator
@exp.run
def train():
    train_model()

# Manual
exp.run()
train_model()
exp.complete()
```

---

## Next Steps

- **API Reference**: See [API.md](API.md) for complete API documentation with examples
- **Architecture**: Read [ARCHITECTURE.md](../src/ml_dash/ARCHITECTURE.md) to understand internals
- **Examples**: Check out [examples/](examples/) directory for more code samples
- **Integration Guide**: See [INTEGRATION_GUIDE.md](../../INTEGRATION_GUIDE.md) for dash-server setup
- **Directory Feature**: See [DIRECTORY_FEATURE.md](../DIRECTORY_FEATURE.md) for detailed directory documentation

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/fortyfive-labs/ml-dash/issues)
- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: Browse [examples/](examples/) for real-world usage patterns

Happy experimenting! 🚀
