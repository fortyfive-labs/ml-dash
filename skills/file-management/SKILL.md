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
  - save_text
  - save_json
  - save_blob
  - model.pth
  - visualization
  - plot
  - matplotlib
---

# ML-Dash File Management

## Fluent File API

### Save Existing File
```python
result = experiment.files("models").save("model.pth")

print(f"Uploaded: {result['filename']}")
print(f"Size: {result['sizeBytes']} bytes")
print(f"Checksum: {result['checksum']}")
```

### Save with Custom Filename
```python
result = experiment.files("data").save(b"hello world", to="greeting.bin")
```

### Save Dict/List as JSON
```python
config = {"model": "resnet50", "lr": 0.001}
result = experiment.files("configs").save(config, to="config.json")

data = [1, 2, 3, {"key": "value"}]
result = experiment.files("data").save(data, to="array.json")
```

## Specialized Save Methods

### Save Text
```python
content = "Hello, World!\nThis is a test."
result = experiment.files("texts").save_text(content, to="greeting.txt")
```

### Save JSON
```python
data = {"hey": "yo", "count": 42}
result = experiment.files("configs").save_json(data, to="config.json")
```

### Save Binary (Blob)
```python
data = b"\x00\x01\x02\x03\x04\x05"
result = experiment.files("data").save_blob(data, to="binary.bin")
```

### Save with Path Including Prefix
```python
# Creates file at /configs/settings.json
result = experiment.files().save({"key": "value"}, to="configs/settings.json")
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

## Listing Files

```python
# List all files
files = experiment.files().list()
for f in files:
    print(f"File: {f['filename']}, Path: {f['path']}, Size: {f['sizeBytes']}")

# List files in a prefix
files = experiment.files("models").list()

# List with glob pattern
json_files = experiment.files("data").list("*.json")
txt_files = experiment.files("data").list("*.txt")
```

## Downloading Files

```python
# Download by file_id
upload_result = experiment.files("models").save("model.pth")
file_id = upload_result["id"]
downloaded = experiment.files(file_id=file_id).download(to="./downloaded_model.pth")

# Download by filename
downloaded = experiment.files("model.txt").download(to="./out.txt")

# Download with glob pattern
downloaded = experiment.files("data").download("*.txt", to="./downloads/")
```

## Deleting Files

```python
# Delete by filename
result = experiment.files("model.txt").delete()

# Delete with pattern
results = experiment.files("data").delete("*.txt")

# Delete using full path
result = experiment.files.delete("models/model.txt")
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
experiment.files("models").save_torch(model, to="model.pt")
experiment.files("models").save_torch(model.state_dict(), to="model_state.pt")
```

## Training Checkpoint Pattern

```python
best_accuracy = 0.0

for epoch in range(100):
    train_loss = train_one_epoch(model)
    val_accuracy = validate(model)

    experiment.metrics("train_loss").log(value=train_loss, epoch=epoch)
    experiment.metrics("val_accuracy").log(value=val_accuracy, epoch=epoch)

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

## Background Uploads

Files are automatically uploaded in the background for non-blocking performance:

```python
with experiment.run:
    # Non-blocking - returns immediately
    experiment.files("models").save_torch(large_model, to="model.pt")

    # Continue training while upload happens in background
    for epoch in range(10):
        train_step()

# All uploads complete before context exits
```
