# ML-Dash Usage Summary

## Installation

```bash
# Install from PyPI
pip install ml-dash

# Or with uv
uv pip install ml-dash
```

## Quick Start

### 1. Basic Experiment Tracking

```python
from ml_dash import Experiment

# Local mode (stores data in local filesystem)
with Experiment(
    prefix="username/my-project/experiment-1",
    dash_root=".dash"  # Local storage directory
).run as exp:
    # Log messages
    exp.log("Training started")

    # Set parameters
    exp.params.set(learning_rate=0.001, batch_size=32, optimizer="Adam")

    # Log metrics
    for epoch in range(10):
        loss = train_one_epoch()
        exp.metrics("train").log(epoch=epoch, loss=loss, accuracy=0.95)

    exp.log("Training completed")
```

### 2. Remote Mode (API Server)

```python
from ml_dash import Experiment

# Remote mode (connects to ML-Dash server)
with Experiment(
    prefix="username/my-project/experiment-1",
    dash_url="https://api.dash.ml"  # or http://localhost:3000
).run as exp:
    exp.log("Experiment running on remote server")
    exp.params.set(lr=0.001)
    exp.metrics("train").log(loss=0.5, step=1)
```

## Template Expansion (New in v0.6.13!)

Use dynamic templates in your prefix:

```python
from ml_dash import Experiment

# Templates: {EXP.name}, {EXP.id}, {EXP.date}
exp = Experiment(
    prefix="username/iclr_2024/{EXP.name}/{EXP.id}",
    dash_root=".dash"
)

# Before starting, set the name
exp.run.name = "my-experiment"

with exp.run:
    # Prefix is now: username/iclr_2024/my-experiment/123456789012345678
    # {EXP.id} auto-generates a unique snowflake ID
    # {EXP.date} would expand to YYYYMMDD format
    exp.log("Template expansion works!")
```

Available template variables:
- `{EXP.name}` - Experiment name
- `{EXP.id}` - Auto-generated 18-digit snowflake ID
- `{EXP.date}` - Date in YYYYMMDD format (e.g., 20260127)
- `{EXP.datetime}` - DateTime in YYYYMMDD.HHMMSS format

## Auto-Start Mode (Simplified)

```python
# Auto-detects prefix from file path and git repo
from ml_dash.auto_start import dxp

# Just use dxp directly - no need to create Experiment
with dxp.run:
    dxp.log("Auto-detected prefix!")
    dxp.params.set(lr=0.01)
    dxp.metrics("train").log(loss=0.3)

# Prefix auto-detected as: {username}/{project}/{year}/{path}/{timestamp}/{job_counter}
```

## Logging

```python
# Simple messages
exp.log("Training started")

# With log levels
exp.log("Debug info", level="debug")
exp.log("Warning message", level="warning")
exp.log("Error occurred", level="error")

# Available levels: debug, info, warning, error, critical
```

## Metrics Tracking

```python
# Single metric
exp.metrics("train").log(loss=0.5, epoch=1)

# Multiple metrics per track
exp.metrics("train").log(loss=0.5, accuracy=0.95, epoch=1)

# Different tracks
exp.metrics("train").log(loss=0.3, epoch=2)
exp.metrics("val").log(loss=0.4, accuracy=0.92, epoch=2)
exp.metrics("test").log(accuracy=0.91)

# Optional metric name (defaults to "default")
exp.metrics().log(loss=0.5)  # Same as exp.metrics("default").log(...)
```

## Parameters

```python
# Set multiple parameters
exp.params.set(
    learning_rate=0.001,
    batch_size=32,
    optimizer="Adam",
    epochs=100
)

# Set nested parameters
exp.params.set(**{
    "model/name": "ResNet-50",
    "model/dropout": 0.3,
    "train/lr": 0.01,
    "train/batch_size": 64
})

# Get parameters (remote mode only)
params = exp.params.get()
print(params["learning_rate"])  # 0.001
```

## File Uploads

```python
# Upload a single file
exp.files.upload("model.pth", "checkpoints/model.pth")

# Upload with metadata
exp.files.upload(
    "results.json",
    "outputs/results.json",
    metadata={"type": "results", "version": "v1"}
)

# Upload multiple files
exp.files.upload("*.log", "logs/")

# Download files (remote mode)
exp.files.download("checkpoints/model.pth", "local_model.pth")

# List files
files = exp.files.list()
for f in files:
    print(f"Name: {f['name']}, Size: {f['size']}")
```

## Video Logging

```python
import numpy as np

# Create video frames (T, H, W, C) or (T, C, H, W)
frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)

# Save as video
exp.save_video(frames, "training_visualization.mp4", fps=30)

# Grayscale video
gray_frames = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
exp.save_video(gray_frames, "grayscale.mp4", fps=25)

# Save as GIF
exp.save_video(frames, "animation.gif", fps=10)
```

## Using RUN Configuration

```python
from ml_dash.run import RUN

# Configure before creating experiment
RUN.owner = "username"
RUN.project = "my-project"
RUN.api_url = "http://localhost:3000"

# These settings will be used by Experiment
from ml_dash.auto_start import dxp
with dxp.run:
    dxp.log("Using RUN configuration")
```

## Sweep Examples

### Running Hyperparameter Sweeps

```bash
# Run a learning rate sweep
cd examples/experiments/sweeps
uv run launch.py --sweep configs/lr_sweep.jsonl --owner username --api-url http://localhost:3000

# Run batch size sweep
uv run launch.py --sweep configs/batch_sweep.jsonl

# Dry run to see commands
uv run launch.py --sweep configs/sweep.jsonl --dry-run
```

### Creating Sweep Training Script

```python
from params_proto import proto
from ml_dash.auto_start import dxp

@proto.prefix
class Train:
    learning_rate: float = 0.01
    batch_size: int = 32
    optimizer: str = "SGD"

@proto.cli
def main():
    # dxp auto-detects configuration
    with dxp.run:
        # Log parameters
        train_params = {k: v for k, v in vars(Train).items()
                       if not k.startswith('_') and not callable(v)}
        dxp.params.set(**{f"train/{k}": v for k, v in train_params.items()})

        # Training loop
        for epoch in range(Train.epochs):
            loss = train_epoch(Train.learning_rate, Train.batch_size)
            dxp.metrics("train").log(epoch=epoch, loss=loss)

if __name__ == "__main__":
    main()
```

## CLI Commands

```bash
# Login to ML-Dash server
ml-dash login

# View profile
ml-dash profile

# List experiments
ml-dash list username/project

# Create experiment/folder
ml-dash create username/project/experiment-1

# Upload local experiments to remote
ml-dash upload .dash --dash-url http://localhost:3000

# Download experiment data
ml-dash download username/project/exp-1 --dest ./downloads

# Remove experiments
ml-dash remove username/project/exp-1
```

## Advanced Features

### Custom Prefix with Templates

```python
from ml_dash import Experiment

# Dynamic prefix based on hyperparameters
exp = Experiment(
    prefix="username/resnet/{EXP.name}/{EXP.date}/{EXP.id}",
    dash_root=".dash"
)

exp.run.name = f"lr{0.01}_bs{32}"
# Expands to: username/resnet/lr0.01_bs32/20260127/123456789012345678

with exp.run:
    exp.log("Custom prefix with templates")
```

### Preventing Prefix Changes After Start

```python
exp = Experiment(prefix="username/project/exp1", dash_root=".dash")

with exp.run:
    # This will raise RuntimeError:
    # exp.run.prefix = "new/prefix"  # ❌ Cannot change after start
    pass
```

### Error Handling

```python
from ml_dash import Experiment

try:
    with Experiment(prefix="user/project/exp", dash_root=".dash").run as exp:
        exp.log("Starting training")
        result = risky_operation()
        exp.metrics("train").log(loss=result)
except Exception as e:
    # Experiment automatically marked as FAILED
    print(f"Experiment failed: {e}")
```

## Configuration

### Environment Variables

```bash
# Set default API URL
export ML_DASH_API_URL="http://localhost:3000"

# Set prefix template
export ML_DASH_PREFIX="username/project/{now:%Y-%m-%d}/exp"
```

### Config File (~/.ml-dash/config.yaml)

```yaml
remote_url: "http://localhost:3000"
api_key: "your-api-key"
default_owner: "username"
```

## Best Practices

1. **Use Templates for Organization**
   ```python
   prefix = "username/{project}/{now:%Y/%m-%d}/{EXP.name}/{EXP.id}"
   ```

2. **Separate Metrics by Track**
   ```python
   exp.metrics("train").log(loss=0.3)
   exp.metrics("val").log(loss=0.4)
   exp.metrics("test").log(accuracy=0.92)
   ```

3. **Log Parameters First**
   ```python
   with exp.run:
       exp.params.set(...)  # Set params immediately
       exp.log("Training started")
       # ... training code
   ```

4. **Use Nested Parameter Names**
   ```python
   exp.params.set(**{
       "model/architecture": "ResNet-50",
       "model/dropout": 0.3,
       "optimizer/type": "Adam",
       "optimizer/lr": 0.001
   })
   ```

5. **Handle Failures Gracefully**
   ```python
   try:
       with exp.run:
           train()
   except Exception as e:
       exp.log(f"Failed: {e}", level="error")
       raise
   ```

## Version Enforcement

ml-dash v0.6.13 enforces strict version checking:

- ✓ Users must be on version 0.6.13 to use the package
- ✓ Older versions are automatically blocked on import
- ✓ Error message guides users to upgrade:

```bash
pip install --upgrade ml-dash
# or
uv pip install --upgrade ml-dash
```

## Links

- **PyPI**: https://pypi.org/project/ml-dash/
- **GitHub**: https://github.com/fortyfive-labs/ml-dash
- **Documentation**: Coming soon
- **Dashboard**: https://dash.ml/

## Support

For issues and questions:
- GitHub Issues: https://github.com/fortyfive-labs/ml-dash/issues
- Documentation: See examples/ directory in the repository
