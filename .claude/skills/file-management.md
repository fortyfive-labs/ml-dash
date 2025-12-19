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
result = experiment.files("model.pth", prefix="/models").save()

print(f"Uploaded: {result['filename']}")
print(f"Size: {result['sizeBytes']} bytes")
print(f"Checksum: {result['checksum']}")
```

## Organizing Files with Prefixes

```python
# Models
experiment.files("model.pth", prefix="/models").save()
experiment.files("best_model.pth", prefix="/models/checkpoints").save()

# Visualizations
experiment.files("loss_curve.png", prefix="/visualizations").save()

# Configuration
experiment.files("config.json", prefix="/config").save()
```

## File Metadata

```python
experiment.files(
    "best_model.pth",
    prefix="/models",
    description="Best model from epoch 50",
    tags=["checkpoint", "best"],
    metadata={
        "epoch": 50,
        "val_accuracy": 0.95
    }
).save()
```

## Downloading Files

```python
# Upload first
upload_result = experiment.files("model.pth", prefix="/models").save()
file_id = upload_result["id"]

# Download to specific path
downloaded = experiment.files(
    file_id=file_id,
    dest_path="./downloaded_model.pth"
).download()

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
    experiment.files(file_id=best["id"], dest_path="./best.pth").download()
```

## Saving Matplotlib Figures

```python
import matplotlib.pyplot as plt

plt.plot([0.5, 0.4, 0.3, 0.2])
plt.title("Training Loss")

# save_fig auto-closes figure
experiment.files(prefix="/visualizations").save_fig("loss_curve.png")

# With custom options
experiment.files(prefix="/visualizations").save_fig(
    "plot.pdf",
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
experiment.files(prefix="/videos").save_video(frames, "animation.mp4")

# Custom FPS
experiment.files(prefix="/videos").save_video(frames, "animation.mp4", fps=30)

# Save as GIF
experiment.files(prefix="/videos").save_video(frames, "animation.gif")
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

dxp.files.save_torch(model, "model.pt")
dxp.files.save_torch(model.state_dict(), "model_state.pt")
```

## Training Checkpoint Pattern

```python
import torch

best_accuracy = 0.0

for epoch in range(100):
    train_loss = train_one_epoch(model)
    val_accuracy = validate(model)

    experiment.metrics("train_loss").append(value=train_loss, epoch=epoch)
    experiment.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch)

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_{epoch+1}.pth")
        experiment.files(
            f"checkpoint_{epoch+1}.pth",
            prefix="/checkpoints",
            tags=["checkpoint"],
            metadata={"epoch": epoch + 1}
        ).save()

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        experiment.files(
            "best_model.pth",
            prefix="/models",
            description=f"Best (acc: {best_accuracy:.4f})",
            tags=["best"],
            metadata={"epoch": epoch + 1, "accuracy": best_accuracy}
        ).save()
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
