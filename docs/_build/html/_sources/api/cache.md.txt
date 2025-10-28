# ml_dash.cache - Local Caching

```{eval-rst}
.. automodule:: ml_dash.cache
   :members:
   :undoc-members:
   :show-inheritance:
```

## LocalCache

Local cache for storing metrics and artifacts when offline or for improved performance.

### Methods

#### `__init__(cache_dir)`

Initialize the cache.

**Parameters:**
- `cache_dir` (str): Directory for cache storage

#### `cache_metric(prefix, metric)`

Cache a metric locally.

**Parameters:**
- `prefix` (str): Experiment prefix
- `metric` (Metric): Metric to cache

#### `cache_params(prefix, params)`

Cache parameters locally.

**Parameters:**
- `prefix` (str): Experiment prefix
- `params` (Dict): Parameters to cache

#### `get_pending_uploads()`

Get all metrics and params pending upload.

**Returns:**
- List of cached items waiting to be uploaded

#### `mark_uploaded(items)`

Mark items as successfully uploaded.

**Parameters:**
- `items` (List): Items to mark as uploaded

#### `clear_uploaded()`

Remove successfully uploaded items from cache.

## Usage Example

```python
from ml_dash.cache import LocalCache

cache = LocalCache(cache_dir="~/.ml-dash/cache")

# Cache is used automatically by MLLogger
# when server is unavailable or for batching
```
