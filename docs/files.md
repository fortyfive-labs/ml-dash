# Files

Upload and manage experiment artifacts - models, plots, configs, and results. Files are automatically checksummed and organized with metadata.

## Basic Upload

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    result = experiment.files("model.pth", prefix="/models")

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
    experiment.files("model.pth", prefix="/models")
    experiment.files("best_model.pth", prefix="/models/checkpoints")

    # Visualizations
    experiment.files("loss_curve.png", prefix="/visualizations")
    experiment.files("confusion_matrix.png", prefix="/visualizations")

    # Configuration
    experiment.files("config.json", prefix="/config")

    # Results
    experiment.files("results.csv", prefix="/results")
```

## Downloading Files

Download files from experiments:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Upload a file first
    upload_result = experiment.files(
        file_path="model.pth",
        prefix="/models"
    ).save()

    file_id = upload_result["id"]

    # Download to specific path
    downloaded_path = experiment.files(
        file_id=file_id,
        dest_path="./downloaded_model.pth"
    ).download()

    print(f"Downloaded to: {downloaded_path}")
```

### Download to Current Directory

If `dest_path` is not specified, the file will be downloaded to the current directory with its original filename:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # List all files
    files = experiment.files().list()

    # Download first file using original filename
    if files:
        file_id = files[0]["id"]
        downloaded_path = experiment.files(file_id=file_id).download()
        print(f"Downloaded: {downloaded_path}")
```

### List and Download

List files and download specific ones:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # List all files
    files = experiment.files().list()

    for file_info in files:
        print(f"File: {file_info['filename']}")
        print(f"  Path: {file_info['path']}")
        print(f"  Size: {file_info['sizeBytes']} bytes")
        print(f"  ID: {file_info['id']}")

    # Download a specific file by ID
    best_model = next(
        (f for f in files if "best" in f.get("tags", [])),
        None
    )

    if best_model:
        downloaded = experiment.files(
            file_id=best_model["id"],
            dest_path="./best_model.pth"
        ).download()
        print(f"Downloaded best model to: {downloaded}")
```

### Checksum Verification

Downloads automatically verify checksums to ensure file integrity:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Upload
    upload_result = experiment.files(
        file_path="model.pth",
        prefix="/models"
    ).save()

    original_checksum = upload_result["checksum"]
    print(f"Original checksum: {original_checksum}")

    # Download (checksum verified automatically)
    downloaded = experiment.files(
        file_id=upload_result["id"],
        dest_path="./verified_model.pth"
    ).download()

    print(f"Download verified and saved to: {downloaded}")
```

## File Metadata

Add description, tags, and custom metadata:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    experiment.files(
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

            experiment.files(
                checkpoint_path,
                prefix="/checkpoints",
                tags=["checkpoint"],
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy}
            )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(model.state_dict(), "best_model.pth")
            experiment.files(
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

Upload matplotlib plots using the convenient `save_fig()` method:

```{code-block} python
:linenos:

import matplotlib.pyplot as plt
import numpy as np
from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Generate plot
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Save directly (auto-closes figure)
    experiment.files(
        prefix="/visualizations",
        description="Training loss over epochs",
        tags=["plot"]
    ).save_fig("loss_curve.png")

    # Save as PDF with custom DPI
    xs = np.linspace(-5, 5, 100)
    plt.plot(xs, np.cos(xs), label='Cosine')
    plt.legend()
    experiment.files(prefix="/visualizations").save_fig(
        "cosine_function.pdf",
        dpi=150,
        transparent=True,
        bbox_inches='tight'
    )
```

**Note:** `save_fig()` automatically closes the figure after saving to prevent memory leaks.

## Saving Videos

Upload video frame stacks using the `save_video()` method:

```{code-block} python
:linenos:

import numpy as np
from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Generate frame stack
    def im(x, y):
        canvas = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                if x - 5 < i < x + 5 and y - 5 < j < y + 5:
                    canvas[i, j] = 1
        return canvas

    frames = [im(100 + i, 80) for i in range(20)]

    # Save as MP4 (default 20 FPS)
    experiment.files(prefix="/videos").save_video(frames, "animation.mp4")

    # Save with custom FPS
    experiment.files(prefix="/videos").save_video(frames, "animation.mp4", fps=30)

    # Save as GIF
    experiment.files(prefix="/videos").save_video(frames, "animation.gif")
```

### Frame Format Support

`save_video()` automatically handles various frame formats:

```{code-block} python
:linenos:

# Grayscale frames (H×W)
frames = [np.random.rand(480, 640) for _ in range(30)]
experiment.files(prefix="/videos").save_video(frames, "grayscale.mp4")

# RGB frames (H×W×3)
frames = [np.random.rand(480, 640, 3) for _ in range(30)]
experiment.files(prefix="/videos").save_video(frames, "rgb.mp4")

# Stacked array (N×H×W or N×H×W×C)
frames = np.random.rand(30, 480, 640, 3)
experiment.files(prefix="/videos").save_video(frames, "stacked.mp4")
```

**Frame value ranges:**
- Float values (0.0 to 1.0) - automatically scaled to 0-255
- Uint8 values (0 to 255) - used directly
- Other formats - automatically converted

### Custom Video Options

Pass additional encoding parameters:

```{code-block} python
:linenos:

# Custom codec and quality
experiment.files(prefix="/videos").save_video(
    frames,
    "high_quality.mp4",
    fps=30,
    codec='libx264',
    quality=8,
    pixelformat='yuv420p'
)

# Animated GIF with custom duration
experiment.files(prefix="/videos").save_video(
    frames,
    "animation.gif",
    fps=10,
    duration=0.1  # 100ms per frame
)
```

**Note:** Video encoding dependencies (`imageio`, `imageio-ffmpeg`, `scikit-image`) are included in the base installation.

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

    experiment.files(
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

The prefix is automatically created from the `prefix` parameter when calling `experiment.files()`. This structure ensures:
- Files with same name in different prefixes don't collide
- Easy organization by file type or purpose
- Unique identification via snowflake IDs

**Remote mode** - Files uploaded to S3, metadata in MongoDB:
- Files stored: `s3://bucket/files/{namespace}/{project}/{experiment}/{prefix}/{file_id}/filename`
- Metadata: path, size, SHA256 checksum, tags, description

**File size limit:** 5GB per file

---

**That's it!** You've completed all the core ML-Dash tutorials. Check out the API Reference for detailed method documentation.
