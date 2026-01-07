# Getting Started

Get started with ML-Dash in under 5 minutes.

## Installation

```bash
pip install ml-dash==0.6.2rc1
```

## Quick Start with Remote Mode

The fastest way to get started is using the pre-configured `dxp` singleton with remote tracking:

### 1. Authenticate

```bash
ml-dash login
```

This opens your browser for secure OAuth2 authentication. Your token is stored securely in your system keychain.

### 2. Start Tracking

```{code-block} python
:linenos:

from ml_dash import dxp

# Start experiment (uploads to https://api.dash.ml by default)
with dxp.run:
    # Log messages
    dxp.log().info("Training started")

    # Track parameters
    dxp.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Track metrics over time
    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # Simulated decreasing loss
        dxp.metrics("train").log(loss=loss, epoch=epoch)

    dxp.log().info("Training completed")
```

That's it! Your experiment is now tracked on the ML-Dash server.

## Local Mode (No Authentication)

Local mode stores everything on your filesystem - perfect for offline work:

```{code-block} python
:linenos:

from ml_dash import Experiment

# Create a experiment (stores data in .dash/ directory)
with Experiment(prefix="my-first-experiment", project="tutorial",
        ).run as experiment:
    # Log messages
    experiment.log().info("Training started")

    # Track parameters
    experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Track metrics over time
    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # Simulated decreasing loss
        experiment.metrics("train").log(loss=loss, epoch=epoch)

    experiment.log().info("Training completed")
```

That's it! Your experiment data is now saved in `.dash/tutorial/my-first-experiment/`.

### Where is My Data?

After running the code above, your data is organized like this:

```
.dash/
└── tutorial/                    # project
    └── my-first-experiment/     # experiment
        ├── logs/
        │   └── logs.jsonl       # your log messages
        ├── parameters/
        │   └── parameters.json  # your hyperparameters
        └── metrics/
            └── train/
                └── data.jsonl   # your metrics
```

## Common Patterns

### Tracking a Training Loop

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="train-model", project="project",
        ).run as experiment:
    # Set hyperparameters
    experiment.params.set(
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
        experiment.metrics.log(
            epoch=epoch,
            train=dict(loss=train_loss, accuracy=val_acc),
            eval=dict(loss=train_loss, accuracy=val_acc)
        )

        # Log important events
        if epoch % 10 == 0:
            experiment.log(f"Checkpoint at epoch {epoch}")
```

### Uploading Files

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="my-experiment", project="project",
        ).run as experiment:
    # Train your model...
    # model.save("model.pth")

    # Upload the model file
    experiment.files("models").save("model.pth")

    # Upload a config file with metadata
    experiment.files("configs").save(
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
    prefix="my-experiment",
    project="team-project",
    remote="https://api.dash.ml"  # token auto-loaded from keychain
).run as experiment:
    # Use exactly the same API as local mode!
    experiment.log().info("Running on remote server")
    experiment.params.set(learning_rate=0.001)
```

The API is identical across local and remote modes!

## Getting Help with Claude Code

If you have [Claude Code](https://claude.com/claude-code) installed, you can get intelligent help directly in your IDE:

```bash
claude-code open
```

Then ask questions like:
- "How do I log parameters from a config class?"
- "Show me an example of tracking metrics during training"
- "How do I upload files with metadata?"

Claude will provide code examples and best practices tailored to ML-Dash, powered by integrated **Claude skills** (.claude/skills/).

---

## Next Steps

- **Learn the basics**: Read the [Overview](overview.md) to understand core concepts
- **Explore features**: Check out guides for [Logging](logging.md), [Parameters](parameters.md), [Metrics](metrics.md), and [Files](files.md)
