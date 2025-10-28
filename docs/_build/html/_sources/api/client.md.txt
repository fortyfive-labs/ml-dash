# ml_dash.client - HTTP Client

```{eval-rst}
.. automodule:: ml_dash.client
   :members:
   :undoc-members:
   :show-inheritance:
```

## DataManagerClient

HTTP client for communicating with the data-manager backend.

### Methods

#### `__init__(server_url)`

Initialize the client.

**Parameters:**
- `server_url` (str): URL of the data-manager server

#### `async post_metrics(prefix, metrics)`

Post metrics to the server.

**Parameters:**
- `prefix` (str): Experiment prefix
- `metrics` (List[Metric]): List of metrics to post

**Returns:**
- Response from server

#### `async post_params(prefix, params)`

Post parameters to the server.

**Parameters:**
- `prefix` (str): Experiment prefix
- `params` (Dict): Parameters to post

**Returns:**
- Response from server

#### `async upload_artifact(prefix, path, data)`

Upload an artifact file.

**Parameters:**
- `prefix` (str): Experiment prefix
- `path` (str): Artifact path
- `data` (bytes): File data

**Returns:**
- Response from server

#### `async download_artifact(prefix, path)`

Download an artifact file.

**Parameters:**
- `prefix` (str): Experiment prefix
- `path` (str): Artifact path

**Returns:**
- File data as bytes
