# ml_logger - Core Logger

```{eval-rst}
.. automodule:: ml_logger
   :members:
   :undoc-members:
   :show-inheritance:
```

## ML_Logger

The main logger class for tracking experiments.

### Methods

#### `__init__(prefix, server_url, backend, **kwargs)`

Initialize a new logger instance.

**Parameters:**
- `prefix` (str): Hierarchical prefix for organizing experiments (e.g., "experiments/project-name")
- `server_url` (str, optional): URL of the data-manager server
- `backend` (str, optional): Storage backend ("local", "s3", etc.)
- `**kwargs`: Additional backend-specific configuration

#### `log_params(**params)`

Log experiment parameters and hyperparameters.

**Parameters:**
- `**params`: Key-value pairs of parameters to log

**Example:**
```python
logger.log_params(
    learning_rate=0.001,
    batch_size=32,
    model="resnet50"
)
```

#### `log_metrics(step=None, **metrics)`

Log training metrics.

**Parameters:**
- `step` (int, optional): Step number (epoch, iteration, etc.)
- `**metrics`: Key-value pairs of metrics to log

**Example:**
```python
logger.log_metrics(
    epoch=10,
    train_loss=0.45,
    val_loss=0.52,
    val_accuracy=0.89
)
```

#### `save(data, path)`

Save arbitrary data as an artifact.

**Parameters:**
- `data`: Data to save
- `path` (str): Relative path within the experiment directory

#### `load(path)`

Load an artifact.

**Parameters:**
- `path` (str): Relative path within the experiment directory

**Returns:**
- Loaded data

#### `save_checkpoint(state_dict, filename)`

Save a model checkpoint.

**Parameters:**
- `state_dict` (dict): Model state dictionary
- `filename` (str): Checkpoint filename

#### `save_figure(fig, filename)`

Save a matplotlib figure.

**Parameters:**
- `fig`: Matplotlib figure object
- `filename` (str): Figure filename
