# Getting Started

Get started with ML-Dash in under 5 minutes.

## Installation

```bash
pip install ml-dash
```

## Your First Experiment


Local mode stores everything on your filesystem - perfect for getting started:

```{code-block} python
:linenos:

from ml_dash import Experiment

# Create a experiment (stores data in .ml-dash/ directory)
with Experiment(name="my-first-experiment", project="tutorial",
        local_path=".ml-dash").run as experiment:
    # Log messages
    experiment.log("Training started")

    # Metric parameters
    experiment.params.set(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )

    # Metric metrics over time
    for epoch in range(10):
        loss = 1.0 - epoch * 0.08  # Simulated decreasing loss
        experiment.metrics("loss").append(value=loss, epoch=epoch)

    experiment.log("Training completed")
```

That's it! Your experiment data is now saved in `.ml-dash/tutorial/my-first-experiment/`.

### Where is My Data?

After running the code above, your data is organized like this:

```
.ml-dash/
└── tutorial/                    # project
    └── my-first-experiment/     # experiment
        ├── logs/
        │   └── logs.jsonl       # your log messages
        ├── parameters/
        │   └── parameters.json  # your hyperparameters
        └── metrics/
            └── loss.jsonl       # your metrics
```

## Common Patterns

### Tracking a Training Loop

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="train-model", project="project",
        local_path=".ml-dash").run as experiment:
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

        # Metric metrics
        experiment.metrics("train_loss").append(value=train_loss, epoch=epoch)
        experiment.metrics("val_accuracy").append(value=val_acc, epoch=epoch)

        # Log important events
        if epoch % 10 == 0:
            experiment.log(f"Checkpoint at epoch {epoch}")
```

### Uploading Files

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Train your model...
    # model.save("model.pth")

    # Upload the model file
    experiment.files(
        file_prefix="model.pth",
        prefix="/models"
    ).save()

    # Upload a config file with metadata
    experiment.files(
        file_prefix="config.yaml",
        prefix="/configs",
        metadata={"version": "1.0"}
    ).save()
```

## Remote Mode

To collaborate with your team, switch to remote mode by pointing to a ML-Dash server:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(
    name="my-experiment",
    project="team-project",
    remote="https://api.dash.ml",
    user_name="your-name"
).run as experiment:
    # Use exactly the same API as local mode!
    experiment.log("Running on remote server")
    experiment.params.set(learning_rate=0.001)
```

The API is identical - just add `remote` and `user_name` parameters.

## Next Steps

- **Learn the basics**: Read the [Overview](overview.md) to understand core concepts
- **Explore features**: Check out guides for [Logging](logging.md), [Parameters](parameters.md), [Metrics](metrics.md), and [Files](files.md)
