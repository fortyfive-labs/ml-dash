# Test Rigs Summary

This document provides an overview of all test rigs created from the quick_start.md documentation.

**Key Examples**:
```python
# Parameter logging
logger.params.log(
    learning_rate=0.001,
    batch_size=32,
    train=dict(epochs=100, early_stopping=True)
)

# Statistics collection
logger.metrics.collect(step=step, train_loss=loss)
logger.metrics.flush(_aggregation="mean", step=epoch)

# Namespacing
logger.metrics("train").log(step=0, loss=0.5)
logger.metrics("val").log(step=0, accuracy=0.85)
```

**Key Examples**:
```python
# Save checkpoint
logger.files("checkpoints").save(model.state_dict(), "model_epoch_10.pt")

# Save images
logger.files.save_image("confusion_matrix", image_ndarray)

# Save video
logger.files.save_video("training_progress", video_ndarray, fps=30)

# Create video from frames
logger.files.make_video(
    "frames/frame_*.png",
    output="video.mp4",
    fps=24,
    sort=True,
)
```

**Key Examples**:
```python
# Hierarchical organization
logger = ML_Logger(
    backend=backend,
    experiment_name="vision/object-detection",
    run_id="yolov8"
)

# Timestamp organization
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = ML_Logger(
    experiment_name="experiments",
    run_id=timestamp
)

# Context managers
with logger.experiment("resnet50"):
    with logger.step(epoch):
        logger.log_metric("loss", 0.5)

# Local logging with upload daemon
logger = ML_Logger(
    prefix="experiments/my-project",
    fs_path="/tmp/ml-logger/data",
)
logger.upload_daemon.start()
```

## Key Testing Patterns

### Pattern 1: Parameter Logging
```python
def test_log_nested_parameters(self, logger):
    logger.params.log(
        learning_rate=0.001,
        train=dict(epochs=100),
        model=dict(layers=50),
    )
    params = logger.params.read()
    assert params["train"]["epochs"] == 100
```

### Pattern 2: Statistics Collection
```python
def test_collect_and_aggregate(self, logger):
    for batch in range(10):
        logger.metrics.collect(step=batch, loss=loss)
    logger.metrics.flush(_aggregation="mean", step=epoch)
```

### Pattern 3: Artifact Management
```python
def test_save_and_load(self, logger):
    logger.files("checkpoints").save(data, "model.pt")
    loaded = logger.files.load_torch("model.pt")
    assert loaded is not None
```

### Pattern 4: Video from Frames
```python
def test_make_video(self, logger):
    for i in range(20):
        logger.files.save_image(f"frames/frame_{i:05d}.png", frame)
    logger.files.make_video("frames/*.png", output="video.mp4", fps=30)
```

### Pattern 5: Authentication Testing (PLANNED FOR LATER PHASE)

```python
def test_env_auth(self, monkeypatch):
    monkeypatch.setenv("ML_LOGGER_TOKEN", "token")
    logger = ML_Logger(backend=backend)
    assert logger is not None
```

---

## Maintenance

### Keeping Tests Updated:
1. When quick_start.md is updated, update corresponding tests
2. Add tests for new features
3. Remove tests for deprecated features
4. Keep fixtures aligned with current API

