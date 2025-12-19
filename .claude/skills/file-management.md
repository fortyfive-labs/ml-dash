---
description: File uploads, downloads, artifacts, checkpoints, visualizations, and videos in ML-Dash
globs:
  - "**/*.py"
  - "**/train*.py"
  - "**/*checkpoint*"
  - "**/*model*"
keywords:
  - files
  - upload
  - download
  - artifact
  - checkpoint
  - save
  - save_fig
  - save_video
  - save_torch
  - model.pth
  - visualization
  - plot
  - matplotlib
---

# ML-Dash File Management

## Basic Upload

```python
result = experiment.files("models").save("model.pth")

print(f"Uploaded: {result['filename']}")
print(f"Size: {result['sizeBytes']} bytes")
print(f"Checksum: {result['checksum']}")
```

## Organizing Files with Paths

```python
# Models
experiment.files("models").save("model.pth")
experiment.files("models/checkpoints").save("best_model.pth")

# Visualizations
experiment.files("visualizations").save("loss_curve.png")

# Configuration
experiment.files("config").save("config.json")
```

## File Metadata

```python
experiment.files("models").save(
    "best_model.pth",
    description="Best model from epoch 50",
    tags=["checkpoint", "best"],
    metadata={
        "epoch": 50,
        "val_accuracy": 0.95
    }
)
```

## Downloading Files

```python
# Upload first
upload_result = experiment.files("models").save("model.pth")
file_id = upload_result["id"]

# Download to specific path
downloaded = experiment.files(file_id=file_id).download(dest_path="./downloaded_model.pth")

# Download with original filename
downloaded = experiment.files(file_id=file_id).download()
```

## List and Download

```python
files = experiment.files().list()

for f in files:
    print(f"File: {f['filename']}, Path: {f['path']}, Size: {f['sizeBytes']}")

# Download specific file
best = next((f for f in files if "best" in f.get("tags", [])), None)
if best:
    experiment.files(file_id=best["id"]).download(dest_path="./best.pth")
```

## Saving Matplotlib Figures

```python
import matplotlib.pyplot as plt

plt.plot([0.5, 0.4, 0.3, 0.2])
plt.title("Training Loss")

# save_fig auto-closes figure
experiment.files("visualizations").save_fig(to="loss_curve.png")

# With custom options
experiment.files("visualizations").save_fig(
    to="plot.pdf",
    dpi=150,
    transparent=True,
    bbox_inches='tight'
)
```

## Saving Videos

```python
import numpy as np

# Generate frames (list of arrays or stacked array)
frames = [np.random.rand(480, 640, 3) for _ in range(30)]

# Save as MP4 (default 20 FPS)
experiment.files("videos").save_video(frames, to="animation.mp4")

# Custom FPS
experiment.files("videos").save_video(frames, to="animation.mp4", fps=30)

# Save as GIF
experiment.files("videos").save_video(frames, to="animation.gif")
```

### Frame Format Support
- Grayscale: (H, W)
- RGB: (H, W, 3)
- Stacked: (N, H, W) or (N, H, W, C)
- Float 0-1 auto-scaled to 0-255

## Saving PyTorch Models

```python
# Using the singleton
from ml_dash import dxp

dxp.files("models").save_torch(model, to="model.pt")
dxp.files("models").save_torch(model.state_dict(), to="model_state.pt")
```

## Training Checkpoint Pattern

```python
best_accuracy = 0.0

for epoch in range(100):
    train_loss = train_one_epoch(model)
    val_accuracy = validate(model)

    experiment.metrics("train_loss").append(value=train_loss, epoch=epoch)
    experiment.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch)

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        experiment.files("checkpoints").save_torch(
            model.state_dict(),
            to=f"checkpoint_{epoch+1}.pth",
            tags=["checkpoint"],
            metadata={"epoch": epoch + 1}
        )

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        experiment.files("models").save_torch(
            model.state_dict(),
            to="best_model.pth",
            description=f"Best (acc: {best_accuracy:.4f})",
            tags=["best"],
            metadata={"epoch": epoch + 1, "accuracy": best_accuracy}
        )
```

## Storage Structure

### Local Mode
```
files/
├── models/
│   └── {snowflake_id}/model.pth
├── checkpoints/
│   └── {snowflake_id}/checkpoint.pth
└── visualizations/
    └── {snowflake_id}/plot.png
```

### Remote Mode
- Files: S3 `s3://bucket/files/{namespace}/{project}/{experiment}/{prefix}/{file_id}/filename`
- Metadata: MongoDB (path, size, SHA256 checksum, tags)
- File size limit: 5GB
