# Image Saving

ML-Dash provides direct support for saving numpy arrays as images in PNG and JPEG formats.

## Overview

The `save_image()` method allows you to save numpy arrays directly without manually converting to image formats. This is perfect for:
- **MuJoCo/PyBullet**: Rendering frames from physics simulations
- **Computer Vision**: Saving model predictions, visualizations
- **Reinforcement Learning**: Saving agent observations
- **Any Numpy Arrays**: Camera feeds, generated images, etc.

## Basic Usage

```python
import numpy as np
from ml_dash import Experiment

with Experiment("vision/training").run as experiment:
    # Get or create a numpy array
    pixels = renderer.render()  # From MuJoCo, OpenCV, etc.

    # Save as PNG (lossless)
    experiment.files("frames").save_image(pixels, to="frame_001.png")

    # Save as JPEG (lossy, smaller files)
    experiment.files("frames").save_image(pixels, to="frame_001.jpg")
```

## Auto-Detection

The `save()` method automatically detects numpy arrays:

```python
# These are equivalent:
experiment.files("images").save(pixels, to="frame.png")
experiment.files("images").save_image(pixels, to="frame.png")
```

## Supported Array Types

### 1. uint8 Arrays (Most Common)

```python
# RGB image (H x W x 3)
pixels_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
experiment.files("images").save_image(pixels_rgb, to="rgb.png")

# Grayscale image (H x W)
pixels_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
experiment.files("images").save_image(pixels_gray, to="gray.png")

# RGBA image with transparency (H x W x 4)
pixels_rgba = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
experiment.files("images").save_image(pixels_rgba, to="rgba.png")
```

### 2. Normalized Float Arrays (0.0 to 1.0)

```python
# Normalized RGB (automatically converted to uint8)
pixels_float = np.random.rand(480, 640, 3)
experiment.files("images").save_image(pixels_float, to="normalized.png")
# Automatically: pixels_float * 255 -> uint8
```

### 3. Arbitrary Range Float Arrays

```python
# Any range - automatically normalized
pixels_arbitrary = np.random.rand(480, 640, 3) * 1000
experiment.files("images").save_image(pixels_arbitrary, to="scaled.png")
# Automatically: (value - min) / (max - min) * 255 -> uint8
```

## Format Support

### PNG (Lossless, Larger Files)

```python
# Supports transparency
experiment.files("images").save_image(rgba_array, to="transparent.png")

# Best for:
# - Screenshots
# - UI elements
# - Exact reproduction needed
# - Images with text/sharp edges
```

### JPEG (Lossy, Smaller Files)

```python
# Default quality (95) - high quality
experiment.files("photos").save_image(pixels, to="photo.jpg")

# Custom quality (1-100)
experiment.files("photos").save_image(pixels, to="low.jpg", quality=60)
experiment.files("photos").save_image(pixels, to="high.jpg", quality=98)

# Both .jpg and .jpeg extensions work
experiment.files("images").save_image(pixels, to="image.jpeg", quality=90)

# Best for:
# - Photos/renders from simulations
# - Large image sequences
# - When file size matters
# - Natural scenes
```

**JPEG Important Notes:**
- Automatically converts RGBA to RGB (alpha channel removed)
- Uses white background when converting transparent images
- Applies optimization for better compression

## Quality Parameter

Control JPEG compression quality (default: 95):

```python
# Maximum quality (largest file)
experiment.files("frames").save_image(pixels, to="frame.jpg", quality=100)

# High quality (default)
experiment.files("frames").save_image(pixels, to="frame.jpg", quality=95)

# Good quality (balanced)
experiment.files("frames").save_image(pixels, to="frame.jpg", quality=85)

# Lower quality (smaller file)
experiment.files("frames").save_image(pixels, to="frame.jpg", quality=70)

# Low quality (very small file)
experiment.files("frames").save_image(pixels, to="frame.jpg", quality=50)
```

**Quality Guidelines:**
- **95-100**: Nearly lossless, large files
- **85-90**: Great for most use cases, good balance
- **70-80**: Visible compression, smaller files
- **50-60**: Noticeable artifacts, very small files
- **Below 50**: Poor quality, not recommended

## MuJoCo Example

```python
import mujoco
import numpy as np
from ml_dash import Experiment

with Experiment("robotics/mujoco-renders").run as experiment:
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    for i in range(1000):
        # Simulate
        mujoco.mj_step(model, data)

        if i % 10 == 0:
            # Render frame
            renderer.update_scene(data)
            pixels = renderer.render()  # numpy array (480, 640, 3)

            # Save as JPEG with good compression
            experiment.files("robot/frames").save_image(
                pixels,
                to=f"frame_{i:05d}.jpg",
                quality=85
            )
```

## OpenCV Example

```python
import cv2
from ml_dash import Experiment

with Experiment("vision/camera-feed").run as experiment:
    cap = cv2.VideoCapture(0)

    for i in range(100):
        ret, frame = cap.read()  # numpy array

        if ret:
            # OpenCV uses BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame
            experiment.files("camera/frames").save_image(
                frame_rgb,
                to=f"frame_{i:04d}.png"
            )

    cap.release()
```

## PIL/Pillow Integration

```python
from PIL import Image
import numpy as np
from ml_dash import Experiment

with Experiment("vision/processing").run as experiment:
    # Load image with PIL
    img = Image.open("input.jpg")

    # Convert to numpy
    img_array = np.array(img)

    # Process...
    processed = some_processing(img_array)

    # Save result
    experiment.files("processed").save_image(
        processed,
        to="output.png"
    )
```

## Saving Image Sequences

```python
with Experiment("video/frames").run as experiment:
    for frame_idx, frame in enumerate(video_frames):
        # Use zero-padded indexing for proper sorting
        experiment.files("video").save_image(
            frame,
            to=f"frame_{frame_idx:05d}.png"
        )
        # Creates: frame_00000.png, frame_00001.png, ...
```

## Buffering and Performance

Image saves are automatically buffered:

```python
with Experiment("high-freq/frames").run as experiment:
    # All saves are non-blocking!
    for i in range(1000):
        experiment.files("frames").save_image(
            frame,
            to=f"frame_{i:05d}.jpg",
            quality=85
        )
        # Returns immediately, queued for background upload

    # All frames uploaded when context exits
```

## Aligning with Tracks

Use consistent step indices:

```python
with Experiment("robotics/tracking").run as experiment:
    for step in range(1000):
        # Save frame with step index
        experiment.files("frames").save_image(
            pixels,
            to=f"frame_{step:05d}.jpg"
        )

        # Save track with same step index
        experiment.track("robot/position").append({
            "step": step,
            "x": x,
            "y": y,
            "z": z
        })

        # Now frames and tracks are aligned!
```

## Different Paths

Organize images in different directories:

```python
with Experiment("training/run1").run as experiment:
    # Raw frames
    experiment.files("frames/raw").save_image(raw_frame, to="frame.png")

    # Processed frames
    experiment.files("frames/processed").save_image(processed, to="frame.png")

    # Visualizations
    experiment.files("viz/predictions").save_image(pred_vis, to="pred.jpg")

    # Debug images
    experiment.files("debug/attention").save_image(attn_map, to="attn.png")
```

## Error Handling

```python
try:
    experiment.files("images").save_image(pixels, to="frame.png")
except ImportError:
    # PIL/Pillow not installed
    print("Please install Pillow: pip install Pillow")
except ValueError as e:
    # Invalid parameters
    print(f"Error: {e}")
```

## Format Comparison

| Aspect | PNG | JPEG |
|--------|-----|------|
| Compression | Lossless | Lossy |
| File Size | Larger | Smaller |
| Transparency | ✓ Yes | ✗ No |
| Quality | Perfect | Configurable |
| Best For | Graphics, text | Photos, renders |
| Speed | Slower | Faster |

## Best Practices

### 1. Choose Format Based on Content

```python
# PNG for graphics/UI
experiment.files("ui").save_image(ui_screenshot, to="screenshot.png")

# JPEG for photos/renders
experiment.files("renders").save_image(render, to="render.jpg", quality=85)
```

### 2. Use Quality 85 for Balanced JPEG

```python
# Good balance of quality and file size
experiment.files("frames").save_image(
    frame,
    to="frame.jpg",
    quality=85  # Recommended for most cases
)
```

### 3. Zero-Pad Frame Numbers

```python
# Good: Proper sorting
for i in range(1000):
    experiment.files("frames").save_image(
        frame,
        to=f"frame_{i:05d}.jpg"
    )
    # frame_00000.jpg, frame_00001.jpg, ...

# Bad: Wrong sorting
for i in range(1000):
    experiment.files("frames").save_image(
        frame,
        to=f"frame_{i}.jpg"
    )
    # frame_1.jpg, frame_10.jpg, frame_2.jpg (wrong order!)
```

### 4. Consistent File Extensions

```python
# Good: All same format
for i in range(100):
    experiment.files("frames").save_image(frame, to=f"frame_{i:05d}.jpg")

# Confusing: Mixed formats
experiment.files("frames").save_image(frame1, to="frame_1.jpg")
experiment.files("frames").save_image(frame2, to="frame_2.png")
```

### 5. Include Frame Info in Metadata

```python
experiment.files("frames",
    description="Training visualization frames",
    tags=["visualization", "epoch_10"]
).save_image(frame, to="frame.jpg")
```

## Requirements

```bash
# Required
pip install Pillow

# Optional (for numpy arrays)
pip install numpy  # Usually already installed for ML
```

## API Reference

### `save_image(array, *, to, quality=95)`

Save numpy array as an image file.

**Parameters:**
- `array` (numpy.ndarray): Image array (HxW or HxWxC)
- `to` (str): Target filename with extension (.png, .jpg, .jpeg)
- `quality` (int, optional): JPEG quality 1-100 (default: 95)

**Returns:**
- Dict with file metadata (or `{"status": "queued"}` if buffered)

**Raises:**
- `ImportError`: If Pillow not installed
- `ValueError`: If `to` parameter missing or invalid array

**Examples:**

```python
# PNG
experiment.files("images").save_image(pixels, to="image.png")

# JPEG with quality
experiment.files("images").save_image(pixels, to="image.jpg", quality=85)

# With path
experiment.files("data/visualizations").save_image(
    vis_array,
    to="prediction.png"
)
```

## Complete Example

```python
import mujoco
import numpy as np
from ml_dash import Experiment

with Experiment(
    prefix="robotics/complete-demo",
    tags=["mujoco", "images", "tracks"],
    dash_url="http://localhost:3000"
).run as experiment:
    experiment.log("Starting robot simulation with image tracking")

    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    steps = 400
    record_interval = 4

    for i in range(steps):
        # Control
        data.ctrl[0] = 4.0 * np.sin(i * 0.03)
        data.ctrl[1] = 2.5 * np.sin(i * 0.05)

        # Simulate
        mujoco.mj_step(model, data)

        # Record
        if i % record_interval == 0:
            # Render
            renderer.update_scene(data)
            pixels = renderer.render()  # numpy array

            # Save image (JPEG for smaller size)
            experiment.files("robot/frames").save_image(
                pixels,
                to=f"frame_{i:05d}.jpg",
                quality=85
            )

            # Track position (aligned with frame)
            ee_pos = data.site_xpos[model.site('end_effector').id].copy()
            experiment.track("robot/position").append({
                "step": i,
                "frame": f"frame_{i:05d}.jpg",  # Reference frame
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2])
            })

            experiment.log(f"Recorded frame_{i:05d}.jpg")

    experiment.log("Simulation complete!")
```

This creates a complete dataset with rendered frames and aligned position tracking!
