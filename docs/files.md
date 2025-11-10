# Files

Upload and manage experiment artifacts - models, plots, configs, and results. Files are automatically checksummed and organized with metadata.

## Basic Upload

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    result = experiment.file("model.pth", prefix="/models")

    print(f"Uploaded: {result['filename']}")
    print(f"Size: {result['sizeBytes']} bytes")
    print(f"Checksum: {result['checksum']}")
```

## Organizing Files

Use paths to organize files logically:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Models
    experiment.file("model.pth", prefix="/models")
    experiment.file("best_model.pth", prefix="/models/checkpoints")

    # Visualizations
    experiment.file("loss_curve.png", prefix="/visualizations")
    experiment.file("confusion_matrix.png", prefix="/visualizations")

    # Configuration
    experiment.file("config.json", prefix="/config")

    # Results
    experiment.file("results.csv", prefix="/results")
```

## File Metadata

Add description, tags, and custom metadata:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.file(
        "best_model.pth",
        prefix="/models",
        description="Best model from epoch 50",
        tags=["checkpoint", "best"],
        metadata={
            "epoch": 50,
            "val_accuracy": 0.95,
            "optimizer_state": True
        }
    )
```

## Training with Checkpoints

Save models during training:

```{code-block} python
:linenos:

import torch
from ml_dash import Experiment

with Experiment(name="resnet-training", project="cv",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(model="resnet50", epochs=100)
    experiment.log("Starting training")

    best_accuracy = 0.0

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Metric metrics
        experiment.metrics("train_loss").append(value=train_loss, epoch=epoch)
        experiment.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)

            experiment.file(
                checkpoint_path,
                prefix="/checkpoints",
                tags=["checkpoint"],
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy}
            )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(model.state_dict(), "best_model.pth")
            experiment.file(
                "best_model.pth",
                prefix="/models",
                description=f"Best model (accuracy: {best_accuracy:.4f})",
                tags=["best"],
                metadata={"epoch": epoch + 1, "accuracy": best_accuracy}
            )

            experiment.log(f"New best model saved (accuracy: {best_accuracy:.4f})")

    experiment.log("Training complete")
```

## Saving Visualizations

Upload matplotlib plots:

```{code-block} python
:linenos:

import matplotlib.pyplot as plt
from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Generate plot
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Save and upload
    plt.savefig("loss_curve.png")
    experiment.file(
        "loss_curve.png",
        prefix="/visualizations",
        description="Training loss over epochs",
        tags=["plot"]
    )

    plt.close()
```

## Uploading Configuration

Save config files alongside parameters:

```{code-block} python
:linenos:

import json
from ml_dash import Experiment

config = {
    "model": {"architecture": "resnet50", "pretrained": True},
    "training": {"epochs": 100, "batch_size": 32, "lr": 0.001}
}

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Metric as parameters
    experiment.params.set(**config)

    # Also save as file
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    experiment.file(
        "config.json",
        prefix="/config",
        description="Experiment configuration",
        tags=["config"]
    )
```

## Storage Format

**Local mode** - Files stored with prefix-based organization:

```
./experiments/
└── project/
    └── my-experiment/
        └── files/
            ├── models/
            │   ├── 7218065541365719/
            │   │   └── model.pth
            │   └── 7218065541366823/
            │       └── best_model.pth
            ├── visualizations/
            │   └── 7218065541367921/
            │       └── loss_curve.png
            └── config/
                └── 7218065541368015/
                    └── config.json
```

Each file is stored as: `files/{prefix}/{snowflake_id}/{filename}`
- **prefix**: Logical organization path (e.g., "models", "configs", "visualizations")
- **snowflake_id**: Unique identifier generated for each file
- **filename**: Original filename

The prefix is automatically created from the `prefix` parameter when calling `experiment.file()`. This structure ensures:
- Files with same name in different prefixes don't collide
- Easy organization by file type or purpose
- Unique identification via snowflake IDs

**Remote mode** - Files uploaded to S3, metadata in MongoDB:
- Files stored: `s3://bucket/files/{namespace}/{project}/{experiment}/{prefix}/{file_id}/filename`
- Metadata: path, size, SHA256 checksum, tags, description

**File size limit:** 5GB per file

---

**That's it!** You've completed all the core ML-Dash tutorials. Check out the API Reference for detailed method documentation.
