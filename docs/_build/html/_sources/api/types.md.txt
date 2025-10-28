# ml_dash.types - Type Interface

```{eval-rst}
.. automodule:: ml_dash.types
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Types

### Experiment

Represents an experiment with metadata and configuration.

**Fields:**
- `id` (str): Unique experiment identifier
- `name` (str): Human-readable experiment name
- `prefix` (str): Hierarchical prefix path
- `created_at` (datetime): Creation timestamp
- `params` (Dict): Experiment parameters
- `tags` (List[str]): Associated tags

### Run

Represents a single training run within an experiment.

**Fields:**
- `id` (str): Unique run identifier
- `experiment_id` (str): Parent experiment ID
- `status` (str): Run status ("running", "completed", "failed")
- `started_at` (datetime): Start timestamp
- `ended_at` (datetime, optional): End timestamp

### Metric

Represents a logged metric value.

**Fields:**
- `name` (str): Metric name
- `value` (float): Metric value
- `step` (int, optional): Step number
- `timestamp` (datetime): Log timestamp

### Artifact

Represents a saved file artifact.

**Fields:**
- `path` (str): Artifact path
- `type` (str): Artifact type ("checkpoint", "figure", "data", etc.)
- `size` (int): File size in bytes
- `created_at` (datetime): Creation timestamp
