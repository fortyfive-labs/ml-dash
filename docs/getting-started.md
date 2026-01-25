# Getting Started

Get started with ML-Dash in under 5 minutes.

## Getting Help with Claude Code

If you have [Claude Code](https://claude.ai/download) installed, you can install the ML-Dash plugin directly from GitHub:

```
/plugin marketplace add fortyfive-labs/ml-dash
```

```
/plugin install ml-dash@ml-dash
```

To update to the latest version:

```
/plugin update ml-dash@ml-dash
```

### Available Skills

Once installed, the following skills are available:

| Skill | Description |
|-------|-------------|
| `cli-commands` | CLI usage for upload, download, list, create |
| `experiment-setup` | Setting up experiments with params-proto |
| `file-management` | Storing and retrieving files and artifacts |
| `getting-started` | Quick start guide |
| `params-proto` | Parameter configuration patterns |
| `tracking-data` | Logging metrics, params, and data |
| `tracks` | Background buffering and track management |

### Ask Questions

With skills loaded, ask questions like:

```console
$ claude "How do I log parameters from a config class?"
Use exp.params.set() or pass a params-proto config to experiment.params.update(Config)
```

```console
$ claude "Show me an example of tracking metrics during training"
Use exp.metrics("train").log(loss=0.5, epoch=1) inside your training loop
```

```console
$ claude "How do I upload experiments to the server?"
Use ml-dash upload -p prefix or set dash_url in your Experiment for auto-sync
```

Claude will provide code examples and best practices tailored to ML-Dash.

---

## Installation

```{parsed-literal}
pip install ml-dash=={{version}}
```

## Quick Start with Remote Mode

The fastest way to get started is using remote tracking with the ML-Dash server:

### 1. Authenticate

```bash
ml-dash login
```

This opens your browser for secure OAuth2 authentication. Your token is stored securely in your system keychain.

### 2. Start Tracking

```{code-block} python
:linenos:

from ml_dash import Experiment

# Start experiment (uploads to https://api.dash.ml by default)
# Prefix format: owner/project/experiment-name
with Experiment(prefix="alice/my-project/training-run").run as exp:
    # Log messages
    exp.log("Training started", level="info")

    # Track parameters
    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Track metrics over time
    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # Simulated decreasing loss
        exp.metrics("train").log(loss=loss, epoch=epoch)

    exp.log("Training completed", level="info")
```

That's it! Your experiment is now tracked on the ML-Dash server.

## Local Mode (No Authentication)

Local mode stores everything on your filesystem - perfect for offline work:

```{code-block} python
:linenos:

from ml_dash import Experiment

# Create an experiment (stores data in .dash/ directory)
# Prefix format: owner/project/experiment-name
with Experiment(
    prefix="alice/tutorial/my-first-experiment",
    dash_root=".dash"
).run as exp:
    # Log messages
    exp.log("Training started", level="info")

    # Track parameters
    exp.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Track metrics over time
    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # Simulated decreasing loss
        exp.metrics("train").log(loss=loss, epoch=epoch)

    exp.log("Training completed", level="info")
```

That's it! Your experiment data is now saved in `.dash/alice/tutorial/my-first-experiment/`.

### Where is My Data?

After running the code above, your data is organized like this:

```
.dash/
└── alice/                              # owner
    └── tutorial/                       # project
        └── my-first-experiment/        # experiment
            ├── logs/
            │   └── logs.jsonl          # your log messages
            ├── parameters/
            │   └── parameters.json     # your hyperparameters
            └── metrics/
                └── train/
                    └── data.jsonl      # your metrics
```

## Common Patterns

### Tracking a Training Loop

```{code-block} python
:linenos:

from ml_dash import Experiment

# Prefix format: owner/project/experiment-name
with Experiment(prefix="alice/project/train-model").run as exp:
    # Set hyperparameters
    exp.params.set(
        model="resnet50",
        optimizer="adam",
        learning_rate=0.001
    )

    # Training loop
    for epoch in range(100):
        # ... your training code ...
        train_loss = 0.5  # your actual loss
        val_acc = 0.9     # your actual accuracy

        # Log metrics
        exp.metrics.log(
            epoch=epoch,
            train=dict(loss=train_loss, accuracy=val_acc),
            eval=dict(loss=train_loss, accuracy=val_acc)
        )

        # Log important events
        if epoch % 10 == 0:
            exp.log(f"Checkpoint at epoch {epoch}")
```

### Uploading Files

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:
    # Train your model...
    # model.save("model.pth")

    # Upload the model file
    exp.files("models").save("model.pth")

    # Upload a config file with metadata
    exp.files("configs").save(
        "config.yaml",
        metadata={"version": "1.0"}
    )
```

## Custom Remote Experiments

Need more control? Create your own experiments with custom names:

```{code-block} python
:linenos:

from ml_dash import Experiment

# First authenticate: ml-dash login

with Experiment(
    prefix="alice/team-project/my-experiment",
    dash_url="https://api.dash.ml"  # token auto-loaded from keychain
).run as exp:
    # Use exactly the same API as local mode!
    exp.log("Running on remote server", level="info")
    exp.params.set(learning_rate=0.001)
```

The API is identical across local and remote modes!

## Next Steps

- **Learn the basics**: Read the [Overview](overview.md) to understand core concepts
- **Explore features**: Check out guides for [Logging](logging.md), [Parameters](parameters.md), [Metrics](metrics.md), and [Files](files.md)
