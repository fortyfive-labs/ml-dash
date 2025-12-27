# Folder (Files)

Upload and manage experiment artifacts - models, plots, configs, and results. Files are automatically checksummed and organized with metadata.

## Fluent Interface Overview

The folder API uses a fluent interface that supports multiple styles:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Upload file from disk
    dxp.files("checkpoints").upload("./model.pt")
    dxp.files("checkpoints").upload("./model.pt", to="checkpoint.pt")

    # Save objects as files
    dxp.files("models").save_torch(model, to="model.pt")
    dxp.files("configs").save_json(config, to="config.json")

    # List files in a location
    files = dxp.files("/models").list()

    # Download a file
    dxp.files("some.text").download()
    dxp.files("some.text").download(to="./local_copy.text")

    # Download using glob patterns
    file_paths = dxp.files("images").list("*.png")
    dxp.files("images").download("*.png", to="local_images")

    # Delete files
    dxp.files("some.text").delete()
    dxp.file.delete("some.text")
    dxp.file.delete("images/*.png")

    # Specific file types
    dxp.file.save_text("content", to="view.yaml")
    dxp.file.save_json(dict(hey="yo"), to="config.json")
    dxp.file.save_blob(b"xxx", to="data.bin")
```

## Basic Upload

### Upload Existing File

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Upload a file to a prefix
    result = dxp.files("models").upload("./model.pth")

    print(f"Uploaded: {result['filename']}")
    print(f"Size: {result['sizeBytes']} bytes")
    print(f"Checksum: {result['checksum']}")
```

### Save Objects as Files

Save Python objects directly without creating intermediate files:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Save dict/list as JSON
    config = {"model": "resnet50", "lr": 0.001}
    dxp.files("configs").save_json(config, to="config.json")

    # Save bytes directly
    dxp.files("data").save_blob(b"binary data", to="data.bin")

    # Save PyTorch model
    import torch
    model = torch.nn.Linear(10, 5)
    dxp.files("checkpoints").save_torch(model, to="checkpoint.pt")
    dxp.files("checkpoints").save_torch(model.state_dict(), to="weights.pt")
```

### Direct Method Style

You can also use the direct method style without specifying a prefix:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Upload file directly
    dxp.file.upload("./model.pt", to="models/model.pt")

    # Save objects directly
    dxp.file.save_text("yaml content", to="configs/view.yaml")
    dxp.file.save_json({"key": "value"}, to="data/config.json")
    dxp.file.save_blob(b"\x00\x01\x02", to="binary/data.bin")
```

## Organizing Files

Use paths to organize files logically:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Models
    dxp.files("models").upload("model.pth")
    dxp.files("models/checkpoints").upload("best_model.pth")

    # Visualizations
    dxp.files("visualizations").upload("loss_curve.png")

    # Configuration
    dxp.files("config").save_json(config, to="config.json")

    # Results
    dxp.files("results").upload("results.csv")
```

## Listing Files

### List All Files

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # List all files
    files = dxp.files().list()

    for file_info in files:
        print(f"File: {file_info['filename']}")
        print(f"  Path: {file_info['path']}")
        print(f"  Size: {file_info['sizeBytes']} bytes")
        print(f"  ID: {file_info['id']}")
```

### List by Prefix

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # List files in specific prefix
    model_files = dxp.files("/models").list()
    config_files = dxp.files("/configs").list()
```

### List with Glob Pattern

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # List files matching pattern
    png_files = dxp.files("images").list("*.png")
    model_files = dxp.files().list("*.pt")
    all_configs = dxp.files().list("**/*.json")
```

## Downloading Files

### Download Single File

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Download by filename/path
    dxp.files("model.pt").download()  # Downloads to current directory
    dxp.files("model.pt").download(to="./local_model.pt")  # Custom destination

    # Download from specific prefix
    dxp.files("models/best.pt").download(to="./best_model.pt")
```

### Download with Glob Pattern

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Download all PNG files from images prefix
    paths = dxp.files("images").download("*.png", to="./local_images")

    # Direct style with path/pattern
    paths = dxp.file.download("images/*.png", to="local_images")

    print(f"Downloaded {len(paths)} files")
```

### Download by File ID (Legacy)

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Upload a file first
    upload_result = dxp.files("models").upload("./model.pth")
    file_id = upload_result["id"]

    # Download by ID
    downloaded_path = dxp.files(file_id=file_id).download()
    print(f"Downloaded to: {downloaded_path}")
```

### Checksum Verification

Downloads automatically verify checksums to ensure file integrity:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Upload
    upload_result = dxp.files("models").upload("./model.pth")
    original_checksum = upload_result["checksum"]
    print(f"Original checksum: {original_checksum}")

    # Download (checksum verified automatically)
    downloaded = dxp.files("model.pth").download(to="./verified_model.pth")
    print(f"Download verified and saved to: {downloaded}")
```

## Deleting Files

### Delete Single File

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Delete by filename/path
    result = dxp.files("some.text").delete()

    # Direct style
    result = dxp.file.delete("some.text")
```

### Delete with Glob Pattern

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Delete all PNG files from images prefix
    results = dxp.files("images").delete("*.png")

    # Direct style
    results = dxp.file.delete("images/*.png")

    print(f"Deleted {len(results)} files")
```

## File Metadata

Add description, tags, and custom metadata:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    result = dxp.files("models").save_torch(
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

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Save text content
    yaml_content = """
    model:
      architecture: resnet50
      pretrained: true
    """
    dxp.files("configs").save_text(yaml_content, to="model.yaml")

    # Or using direct style
    dxp.file.save_text(yaml_content, to="configs/model.yaml")
```

### Save JSON

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    config = {"model": "resnet50", "lr": 0.001}

    # Save JSON
    dxp.files("configs").save_json(config, to="config.json")

    # Or direct style
    dxp.file.save_json(config, to="configs/training.json")
```

### Save Binary Data

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    binary_data = b"\x00\x01\x02\x03"

    # Save blob
    dxp.files("data").save_blob(binary_data, to="weights.bin")

    # Or direct style
    dxp.folder.save_blob(binary_data, to="data/embeddings.bin")
```

## Training with Checkpoints

Save models during training:

```{code-block} python
:linenos:

import torch
from ml_dash import Experiment

with Experiment(name="resnet-training", project="cv",
        local_path=".ml-dash").run as dxp:

    dxp.params.set(model="resnet50", epochs=100)
    dxp.log("Starting training")

    best_accuracy = 0.0

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Log metrics
        dxp.metrics("train_loss").append(value=train_loss, epoch=epoch)
        dxp.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            dxp.files("checkpoints").save_torch(
                model.state_dict(),
                to=f"checkpoint_epoch_{epoch + 1}.pt",
                tags=["checkpoint"],
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy}
            )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            dxp.files("models").save_torch(
                model.state_dict(),
                to="best_model.pt",
                description=f"Best model (accuracy: {best_accuracy:.4f})",
                tags=["best"],
                metadata={"epoch": epoch + 1, "accuracy": best_accuracy}
            )

            dxp.log(f"New best model saved (accuracy: {best_accuracy:.4f})")

    dxp.log("Training complete")
```

## Saving Visualizations

Upload matplotlib plots using the convenient `save_fig()` method:

```{code-block} python
:linenos:

import matplotlib.pyplot as plt
import numpy as np
from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as dxp:

    # Generate plot
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Save directly (auto-closes figure)
    dxp.files("visualizations").save_fig(to="loss_curve.png")

    # Save as PDF with custom DPI
    xs = np.linspace(-5, 5, 100)
    plt.plot(xs, np.cos(xs), label='Cosine')
    plt.legend()
    dxp.files("visualizations").save_fig(
        to="cosine_function.pdf",
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
        local_path=".ml-dash").run as dxp:

    # Generate frame stack
    frames = [np.random.rand(200, 200) for _ in range(20)]

    # Save as MP4 (default 20 FPS)
    dxp.files("videos").save_video(frames, to="animation.mp4")

    # Save with custom FPS
    dxp.files("videos").save_video(frames, to="animation.mp4", fps=30)

    # Save as GIF
    dxp.files("videos").save_video(frames, to="animation.gif")
```

### Frame Format Support

`save_video()` automatically handles various frame formats:

```{code-block} python
:linenos:

# Grayscale frames (H×W)
frames = [np.random.rand(480, 640) for _ in range(30)]
dxp.files("videos").save_video(frames, to="grayscale.mp4")

# RGB frames (H×W×3)
frames = [np.random.rand(480, 640, 3) for _ in range(30)]
dxp.files("videos").save_video(frames, to="rgb.mp4")

# Stacked array (N×H×W or N×H×W×C)
frames = np.random.rand(30, 480, 640, 3)
dxp.files("videos").save_video(frames, to="stacked.mp4")
```

**Frame value ranges:**
- Float values (0.0 to 1.0) - automatically scaled to 0-255
- Uint8 values (0 to 255) - used directly
- Other formats - automatically converted

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

**Remote mode** - Files uploaded to S3, metadata in MongoDB:
- Files stored: `s3://bucket/files/{namespace}/{project}/{experiment}/{prefix}/{file_id}/filename`
- Metadata: path, size, SHA256 checksum, tags, description

**File size limit:** 100GB per file

---

**That's it!** You've completed all the core ML-Dash tutorials. Check out the API Reference for detailed method documentation.
