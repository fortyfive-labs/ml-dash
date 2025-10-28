# ML-Logger Architecture

## Class Hierarchy and Composition

<details open>
<summary><strong>🏗️ System Overview</strong></summary>

```
ML-Logger System
│
├── Storage Backends (remove existing implementations awaiting design)
    add local logger, s3, gcp, ml_dash, as empty files. Also include an empty base class.
├── Logger Components (file and data types)  
├── ML_Logger (Main Interface)
└── Supporting and Utility Classes
```

</details>

<details>
<summary><strong>💾 Storage Backends</strong> (Where to store)</summary>

```
Storage Backends
│
├── StorageBackend (Abstract Base)
│   ├── exists()
│   ├── write_bytes()
│   ├── read_bytes()
│   ├── write_text()
│   ├── read_text()
│   ├── append_text()
│   ├── list_dir()
│   └── get_url()
│
├── LocalBackend(StorageBackend)
│   └── Implements file system operations
│
├── S3Backend(StorageBackend)
│   └── Implements AWS S3 operations
│
└── GCPBackend(StorageBackend)
    └── Implements Google Cloud Storage operations
```

</details>

<details>
<summary><strong>📝 Logger Components</strong> (What to log)</summary>

```
Experiment
│
├── logs: TextLogger
│   ├── log(level, message)
│   ├── error(message)
│   ├── warning(message)
│   ├── info(message)
│   └── debug(message)
│
├── metrics: ScalarLogger (accessed via experiment.metrics)
│   ├── log(step, **metrics) - Log metrics immediately
│   ├── collect(step, **metrics) - Collect for later aggregation
│   ├── flush(_aggregation, step) - Aggregate and log collected metrics
│   ├── get_summary(name, frequency)
│   ├── __call__(namespace) - Return namespaced logger
│   └── Uses: ScalarCache, Series
│
├── files: ArtifactLogger (accessed via experiment.files)
│   ├── save(data, filename) - Save generic data
│   ├── save_pkl(data, filename) - Save pickle data
│   ├── save_image(name, image) - Save image
│   ├── save_video(name, video, fps) - Save video
│   ├── save_audio(name, audio) - Save audio
│   ├── savefig(fig, filename) - Save matplotlib figure
│   ├── load_torch(filename) - Load PyTorch data
│   ├── make_video(pattern, output, fps, codec, quality, sort) - Create video from frames
│   ├── __call__(namespace) - Return namespaced logger
│   └── File management and artifact storage
│
├── params: ParameterIndex
│   ├── set(params) - Set/overwrite parameters
│   ├── extend(params) - Merge with existing parameters
│   ├── update(key, value) - Update single parameter
│   ├── read() - Read all parameters
│   └── Manages experiment configuration
│
└── charts: ChartBuilder  # PLANNING PHASE, subject to changes.
    ├── line_chart(query)
    ├── scatter_plot(query)
    ├── bar_chart(query)
    └── video/images(query)
```

</details>

<details>
<summary><strong>🎯 Composite Logger</strong> (Main Interface)</summary>

```
MLLogger
├── __init__(backend: StorageBackend)
├── params: ParameterIndex - Parameter management
├── metrics: ScalarLogger - Metrics logging
├── readme: MarkdownLogger - Rich Text logging (PLANNING PHASE)
├── files: ArtifactLogger - File and artifact management
├── logs: TextLogger - Text logging
│
├── Convenience Methods: (can just hide under logs)
│   ├── error() -> text.error()
│   ├── warning() -> text.warning()
│   ├── info() -> text.info()
│   └── debug() -> text.debug()
│
└── Context Managers:
    ├── experiment(name)
    └── run(id)
```

</details>

<details>
<summary><strong>⚙️ Supporting Classes</strong></summary>

```
Supporting Classes
│
└── Serialization (serdes/) (NOT USED)
    ├── serialize()
    ├── deserialize()
    └── Type registry with $t, $s keys
```

</details>

## Usage Examples

<details>
<summary><strong>📊 Logging Different Data Types</strong></summary>

```python
# Text logging (errors, warnings, info) experiment.logs.error("Training failed") experiment.logs.warning("Low GPU memory") experiment.logs.info("Starting epoch 1")

# Parameter logging experiment.params.set(learning_rate=0.001, batch_size=32)

# Metrics logging experiment.metrics.log(step=100, loss=0.523, accuracy=0.95)

# Collect metrics for aggregation experiment.metrics.collect(step=101, loss=0.521) experiment.metrics.flush(_aggregation="mean", step=100)

# Namespaced metrics experiment.metrics("train").log(step=100, loss=0.5) experiment.metrics("val").log(step=100, accuracy=0.95)

# File operations experiment.files.save_image("confusion_matrix", image_array) experiment.files.save(model_state, "checkpoint.pt") experiment.files("checkpoints").save(model_state, "model_epoch_10.pt")
```

</details>

<details>
<summary><strong>🎛️ Direct Component Access</strong></summary>

```python
# Access components directly for advanced usage experiment.logs.error("Direct text logging") experiment.metrics.log(step=50, lr=0.001) experiment.files.save_video("training_progress", video_array, fps=30)

# Namespaced file operations experiment.files("videos").save_video("training_progress", video_array, fps=30) experiment.files("checkpoints").save(model_state, "model.pt")

# Get statistics
stats = experiment.metrics.get_stats("loss")
percentile_95 = experiment.metrics.get_percentile("loss", 95)
```

</details>

## File Organization

<details>
<summary><strong>📁 Project Structure</strong></summary>

```
ml-logger/
├── src/ml_logger/
│   ├── __init__.py
│   ├── experiment.py          # Main MLLogger class
│   │
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py        # StorageBackend ABC
│   │   ├── local.py       # LocalBackend
│   │   ├── s3.py          # S3Backend
│   │   └── gcp.py         # GCPBackend
│   │
│   ├── loggers/
│   │   ├── __init__.py
│   │   ├── text.py        # TextLogger
│   │   ├── scalar.py      # ScalarLogger
│   │   └── artifact.py    # ArtifactLogger
│   │
│   ├── scalar_cache.py    # ScalarCache, Series, RollingStats
│   │
│   └── serdes/
│       ├── __init__.py
│       └── ndjson.py      # Serialization with $t, $s
│
└── tests/
    ├── test_backends.py
    ├── test_loggers.py
    ├── test_scalar_cache.py
    └── test_integration.py
```

</details>

## Advanced Features

<details>
<summary><strong>📈 Statistical Features</strong></summary>

### Rolling Statistics
- **Window-based metrics**: Configurable window size for recent data
- **Automatic calculation**: Mean, variance, std, min, max
- **Percentiles**: p0, p1, p5, p10, p20, p25, p40, p50, p60, p75, p80, p90, p95, p99, p100

### Summary Frequencies
Automatic summaries at: 1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 1000, 1200, 1500, 2000, 2500, ...

```python
# Access statistics
stats = experiment.scalars.get_stats("loss")
print(f"Mean: {stats.mean}, Std: {stats.std}")

# Get percentiles
p95 = experiment.scalars.get_percentile("accuracy", 95)

# Get summaries at specific frequencies
summaries = experiment.scalars.get_summary("loss", frequency=100)
```

</details>

<details>
<summary><strong>🔄 Serialization System</strong></summary>

### Type-Annotated Serialization
- Uses `$t` for type keys
- Uses `$s` for shape keys (arrays)
- Recursive serialization for nested structures
- Supports: primitives, datetime, numpy, Path, bytes, collections

```python
from ml_dash.serdes import serialize, deserialize

# Serialize complex objects
data = {
    "array": np.array([[1, 2], [3, 4]]),
    "date": datetime.now(),
    "path": Path("/tmp/file.txt")
}
serialized = serialize(data)

# Deserialize back
original = deserialize(serialized)
```

</details>

## Examples

<details>
<summary><strong>🤖 ML Training Example</strong></summary>

```python
# train.py - Define your training function
from ml_dash import get_logger


@logger.run
def train(config):
    """Training function that will be wrapped by the experiment."""
    model = create_model(config.model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_acc = 0
    for epoch in range(config.epochs):
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_step(model, data, target, optimizer)

            step = epoch * len(train_loader) + batch_idx
            with experiment.step(step):
                # Log metrics
                experiment.log_metric("train/loss", loss.item())

                # Log histograms periodically
                if step % 100 == 0:
                    experiment.log_histogram("gradients", get_gradients(model))

                # Save visualizations
                if step % 500 == 0:
                    fig = plot_predictions(model, data)
                    experiment.log_image("predictions", fig)

        # Validation
        val_loss, val_acc = validate(model, val_loader)
        experiment.log_metrics({
            "val/loss": val_loss,
            "val/accuracy": val_acc
        }, step=epoch)

        # Save checkpoint
        if val_acc > best_acc:
            experiment.log_model("best_model", model.state_dict())
            best_acc = val_acc

    # Final summary
    experiment.info(f"Training completed. Best accuracy: {best_acc}")
    return {"best_accuracy": best_acc}

```

**experiment.py** - Launch experiments with different configs:

```python
from ml_dash import get_logger
from train import train

# Initialize logger
experiment = get_logger("s3://experiments/mnist")

# Define experiment configurations
configs = [
    {"model_type": "CNN", "lr": 0.001, "batch_size": 32, "epochs": 100},
    {"model_type": "CNN", "lr": 0.01, "batch_size": 64, "epochs": 100},
    {"model_type": "ResNet", "lr": 0.001, "batch_size": 32, "epochs": 150},
]

# Run experiment with multiple configurations
with experiment.experiment("model_comparison"):
    for i, config in enumerate(configs):
        # Each config gets its own run
        run_name = f"{config['model_type']}_lr{config['lr']}"

        # The decorator handles run creation and lifecycle
        result = train(
            config=config,
            _run_name=run_name,
            _hyperparams=config,
            _tags=["baseline", config["model_type"].lower()]
        )

        print(f"Run {run_name} completed with accuracy: {result['best_accuracy']}")
```

</details>

<details>
<summary><strong>🔍 Debugging Example</strong></summary>

```python
# Setup logger with debug level
 experiment =get_logger("./debug_logs") experiment.logs.set_level(LogLevel.DEBUG)

try:
    # Your code here
    result = risky_operation()
    experiment.debug(f"Operation result: {result}")
    
except Exception as e:
    # Log exception with full traceback
    experiment.exception("Operation failed", exc_info=True)
    
    # Log additional context
    experiment.error("Failed at step", step=current_step, 
                input_shape=data.shape)
    
    # Save problematic data for debugging
    experiment.log_file("failed_input", "debug_data.pkl")
    
finally:
    # Get recent logs
    errors = experiment.get_logs(level="ERROR", limit=50)
    print(f"Found {len(errors)} errors")
```

</details>