# Folder (Files)

Upload and manage experiment artifacts - models, plots, configs, and results. Files are automatically checksummed and organized with metadata.

## Fluent Interface Overview

The folder API uses a fluent interface that supports multiple styles:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Upload file from disk
    exp.files("checkpoints").upload("./model.pt")
    exp.files("checkpoints").upload("./model.pt", to="checkpoint.pt")

    # Save objects as files
    exp.files("models").save_torch(model, to="model.pt")
    exp.files("configs").save_json(config, to="config.json")

    # List files in a location
    files = exp.files("/models").list()

    # Download a file
    exp.files("some.text").download()
    exp.files("some.text").download(to="./local_copy.text")

    # Download using glob patterns
    file_paths = exp.files("images").list("*.png")
    exp.files("images").download("*.png", to="local_images")

    # Delete files
    exp.files("some.text").delete()
    exp.file.delete("some.text")
    exp.file.delete("images/*.png")

    # Specific file types
    exp.file.save_text("content", to="view.yaml")
    exp.file.save_json(dict(hey="yo"), to="config.json")
    exp.file.save_blob(b"xxx", to="data.bin")
```

## Basic Upload

### Upload Existing File

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Upload a file to a prefix
    result = exp.files("models").upload("./model.pth")

    print(f"Uploaded: {result['filename']}")
    print(f"Size: {result['sizeBytes']} bytes")
    print(f"Checksum: {result['checksum']}")
```

### Save Objects as Files

Save Python objects directly without creating intermediate files:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Save dict/list as JSON
    config = {"model": "resnet50", "lr": 0.001}
    exp.files("configs").save_json(config, to="config.json")

    # Save bytes directly
    exp.files("data").save_blob(b"binary data", to="data.bin")

    # Save PyTorch model
    import torch
    model = torch.nn.Linear(10, 5)
    exp.files("checkpoints").save_torch(model, to="checkpoint.pt")
    exp.files("checkpoints").save_torch(model.state_dict(), to="weights.pt")
```

### Direct Method Style

You can also use the direct method style without specifying a prefix:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Upload file directly
    exp.files.upload("./model.pt", to="models/model.pt")

    # Save objects directly
    exp.files.save_text("yaml content", to="configs/view.yaml")
    exp.files.save_json({"key": "value"}, to="data/config.json")
    exp.files.save_blob(b"\x00\x01\x02", to="binary/data.bin")
```

## Organizing Files

Use paths to organize files logically:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Models
    exp.files("models").upload("model.pth")
    exp.files("models/checkpoints").upload("best_model.pth")

    # Visualizations
    exp.files("visualizations").upload("loss_curve.png")

    # Configuration
    exp.files("config").save_json(config, to="config.json")

    # Results
    exp.files("results").upload("results.csv")
```

## Listing Files

### List All Files

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # List all files
    files = exp.files().list()

    for file_info in files:
        print(f"File: {file_info['filename']}")
        print(f"  Path: {file_info['path']}")
        print(f"  Size: {file_info['sizeBytes']} bytes")
        print(f"  ID: {file_info['id']}")
```

### List by Prefix

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # List files in specific prefix
    model_files = exp.files("/models").list()
    config_files = exp.files("/configs").list()
```

### List with Glob Pattern

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # List files matching pattern
    png_files = exp.files("images").list("*.png")
    model_files = exp.files().list("*.pt")
    all_configs = exp.files().list("**/*.json")
```

## Downloading Files

### Download Single File

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Download by filename/path
    exp.files("model.pt").download()  # Downloads to current directory
    exp.files("model.pt").download(to="./local_model.pt")  # Custom destination

    # Download from specific prefix
    exp.files("models/best.pt").download(to="./best_model.pt")
```

### Download with Glob Pattern

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Download all PNG files from images prefix
    paths = exp.files("images").download("*.png", to="./local_images")

    # Direct style with path/pattern
    paths = exp.file.download("images/*.png", to="local_images")

    print(f"Downloaded {len(paths)} files")
```

### Download by File ID (Legacy)

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Upload a file first
    upload_result = exp.files("models").upload("./model.pth")
    file_id = upload_result["id"]

    # Download by ID
    downloaded_path = exp.files(file_id=file_id).download()
    print(f"Downloaded to: {downloaded_path}")
```

### Checksum Verification

Downloads automatically verify checksums to ensure file integrity:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Upload
    upload_result = exp.files("models").upload("./model.pth")
    original_checksum = upload_result["checksum"]
    print(f"Original checksum: {original_checksum}")

    # Download (checksum verified automatically)
    downloaded = exp.files("model.pth").download(to="./verified_model.pth")
    print(f"Download verified and saved to: {downloaded}")
```

## Deleting Files

### Delete Single File

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Delete by filename/path
    result = exp.files("some.text").delete()

    # Direct style
    result = exp.file.delete("some.text")
```

### Delete with Glob Pattern

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Delete all PNG files from images prefix
    results = exp.files("images").delete("*.png")

    # Direct style
    results = exp.file.delete("images/*.png")

    print(f"Deleted {len(results)} files")
```

## File Metadata

Add description, tags, and custom metadata:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    result = exp.files("models").save_torch(
        model,
        to="best_model.pth",
        description="Best model from epoch 50",
        tags=["checkpoint", "best"],
        metadata={
            "epoch": 50,
            "val_accuracy": 0.95,
            "optimizer_state": True
        }
    )
```

## Saving Specific File Types

### Save Text

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Save text content
    yaml_content = """
    model:
      architecture: resnet50
      pretrained: true
    """
    exp.files("configs").save_text(yaml_content, to="model.yaml")

    # Or using direct style
    exp.file.save_text(yaml_content, to="configs/model.yaml")
```

### Save JSON

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    config = {"model": "resnet50", "lr": 0.001}

    # Save JSON
    exp.files("configs").save_json(config, to="config.json")

    # Or direct style
    exp.file.save_json(config, to="configs/training.json")
```

### Save Binary Data

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    binary_data = b"\x00\x01\x02\x03"

    # Save blob
    exp.files("data").save_blob(binary_data, to="weights.bin")

    # Or direct style
    exp.folder.save_blob(binary_data, to="data/embeddings.bin")
```

## Training with Checkpoints

Save models during training:

```{code-block} python
:linenos:

import torch
from ml_dash import Experiment

with Experiment(prefix="alice/cv/resnet-training").run as exp:

    exp.params.set(model="resnet50", epochs=100)
    exp.log("Starting training")

    best_accuracy = 0.0

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Log metrics (single call with nested dict)
        exp.metrics.log(
            epoch=epoch,
            train=dict(loss=train_loss, accuracy=val_accuracy),
            eval=dict(loss=val_loss, accuracy=val_accuracy)
        )

        # Alternative: prefix-based logging
        # exp.metrics("train").log(loss=train_loss, accuracy=val_accuracy)
        # exp.metrics("eval").log(loss=val_loss, accuracy=val_accuracy)
        # exp.metrics.log(epoch=epoch).flush()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            exp.files("checkpoints").save_torch(
                model.state_dict(),
                to=f"checkpoint_epoch_{epoch + 1}.pt",
                tags=["checkpoint"],
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy}
            )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            exp.files("models").save_torch(
                model.state_dict(),
                to="best_model.pt",
                description=f"Best model (accuracy: {best_accuracy:.4f})",
                tags=["best"],
                metadata={"epoch": epoch + 1, "accuracy": best_accuracy}
            )

            exp.log(f"New best model saved (accuracy: {best_accuracy:.4f})")

    exp.log("Training complete")
```

## Saving Visualizations

Upload matplotlib plots using the convenient `save_fig()` method:

```{code-block} python
:linenos:

import matplotlib.pyplot as plt
import numpy as np
from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Generate plot
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Save directly (auto-closes figure)
    exp.files("visualizations").save_fig(to="loss_curve.png")

    # Save as PDF with custom DPI
    xs = np.linspace(-5, 5, 100)
    plt.plot(xs, np.cos(xs), label='Cosine')
    plt.legend()
    exp.files("visualizations").save_fig(
        to="cosine_function.pdf",
        dpi=150,
        transparent=True,
        bbox_inches='tight'
    )
```

**Note:** `save_fig()` automatically closes the figure after saving to prevent memory leaks.

## Saving Videos

Upload video frame stacks using the `save_video()` method. This is useful for saving training visualizations, agent rollouts, or any sequence of images.

```{code-block} python
:linenos:

import numpy as np
from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Generate frame stack
    frames = [np.random.rand(200, 200) for _ in range(20)]

    # Save as MP4 (default 20 FPS)
    exp.files("videos").save_video(frames, to="animation.mp4")

    # Save with custom FPS
    exp.files("videos").save_video(frames, to="animation.mp4", fps=30)

    # Save as GIF
    exp.files("videos").save_video(frames, to="animation.gif")
```

### Practical Example: Agent Rollout

Record an agent's trajectory or any animated visualization:

```{code-block} python
:linenos:

import numpy as np
from ml_dash import Experiment

def render_frame(x, y):
    """Render a frame with a marker at position (x, y)."""
    canvas = np.zeros((200, 200))
    # Draw a 10x10 square at position
    canvas[max(0, x-5):x+5, max(0, y-5):y+5] = 1.0
    return canvas

with Experiment(prefix="alice/rl/agent-rollout").run as exp:

    # Simulate agent moving across the canvas
    frames = [render_frame(100 + i, 80) for i in range(20)]

    result = exp.files("videos").save_video(frames, to="rollout.mp4")
    print(f"Saved: {result['filename']} ({result['sizeBytes']} bytes)")
```

### Video Encoding Options

Control video quality and encoding with additional parameters:

```{code-block} python
:linenos:

# High quality MP4
exp.files("videos").save_video(frames, to="high_quality.mp4", fps=30, quality=8)

# Lower quality for smaller file size
exp.files("videos").save_video(frames, to="compressed.mp4", fps=30, quality=5)
```

Additional keyword arguments are passed to imageio's writer (e.g., `quality`, `codec`, `bitrate`).

### Frame Format Support

`save_video()` automatically handles various frame formats:

```{code-block} python
:linenos:

# Grayscale frames (H×W)
frames = [np.random.rand(480, 640) for _ in range(30)]
exp.files("videos").save_video(frames, to="grayscale.mp4")

# RGB frames (H×W×3)
frames = [np.random.rand(480, 640, 3) for _ in range(30)]
exp.files("videos").save_video(frames, to="rgb.mp4")

# Stacked array (N×H×W or N×H×W×C)
frames = np.random.rand(30, 480, 640, 3)
exp.files("videos").save_video(frames, to="stacked.mp4")
```

**Frame value ranges:**
- Float values (0.0 to 1.0) - automatically scaled to 0-255
- Uint8 values (0 to 255) - used directly
- Float32 values - automatically converted

**Note:** An empty frame list raises `ValueError: frame_stack is empty`.

## Storage Format

**Local mode** - Files stored with prefix-based organization:

```
.dash/
└── alice/                              # owner
    └── project/                        # project
        └── my-experiment/              # experiment name
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

**Remote mode** - Files uploaded to S3, metadata in MongoDB:
- Files stored: `s3://bucket/files/{namespace}/{project}/{experiment}/{prefix}/{file_id}/filename`
- Metadata: path, size, SHA256 checksum, tags, description

**File size limit:** 100GB per file

---

**That's it!** You've completed all the core ML-Dash tutorials. Check out the API Reference for detailed method documentation.
