# Breaking Changes - ML-Dash Python Client

**Version**: Next Release (TBD)
**Date**: 2026-01-16

This document outlines breaking changes made to the ML-Dash Python client to align with the new Unified Node API on the server.

---

## Summary

The Python client has been updated to use the new Unified Node API endpoints. This is a **BREAKING CHANGE** that requires updates to your code.

### Why These Changes?

- Server deprecated old REST API endpoints on 2026-01-16
- New unified API provides better consistency and hierarchy support
- Improved performance and reduced API surface area

---

## Breaking Changes

### 1. RemoteClient Initialization - **BREAKING**

**Before:**
```python
from ml_dash.client import RemoteClient

client = RemoteClient(
    base_url="http://localhost:3000",
    api_key="your-token"
)
```

**After:**
```python
from ml_dash.client import RemoteClient

client = RemoteClient(
    base_url="http://localhost:3000",
    namespace="your-namespace",  # NEW REQUIRED PARAMETER
    api_key="your-token"
)
```

**Impact**: All code creating `RemoteClient` must be updated.

---

### 2. File Upload - Prefix Parameter Deprecated

**Before:**
```python
client.upload_file(
    experiment_id="exp-123",
    file_path="./model.pt",
    prefix="/models/checkpoints",  # Used for hierarchy
    filename="model.pt",
    ...
)
```

**After:**
```python
client.upload_file(
    experiment_id="exp-123",
    file_path="./model.pt",
    prefix="/models/checkpoints",  # DEPRECATED - ignored
    filename="model.pt",
    parent_id="folder-node-id",  # NEW - use folder node IDs
    ...
)
```

**Impact**: File organization now uses explicit folder nodes instead of string prefixes.

**Migration**:
- For root-level files: Use `parent_id="ROOT"` (default)
- For files in folders: Create folder nodes first, then use their IDs

---

### 3. File Listing - Prefix Filtering Not Supported

**Before:**
```python
# List files with prefix filter
files = client.list_files(
    experiment_id="exp-123",
    prefix="/models"  # Server-side filtering
)
```

**After:**
```python
# List all files (prefix filtering removed)
files = client.list_files(
    experiment_id="exp-123",
    prefix="/models"  # DEPRECATED - ignored
)

# Client-side filtering if needed
filtered = [f for f in files if f["pPath"].startswith("/models")]
```

**Impact**: Prefix filtering must be done client-side. Tag filtering still works.

---

### 4. Response Format Changes

#### Experiment Creation

**Before:**
```python
result = client.create_or_update_experiment(...)
# result = {
#     "experiment": {...},
#     "project": {...},
#     "namespace": {...}
# }
```

**After:**
```python
result = client.create_or_update_experiment(...)
# result = {
#     "node": {...},        # NEW - node record
#     "experiment": {...},  # Same as before
#     "project": {...}      # Same as before
# }
```

**Impact**: Response now includes `node` object. `namespace` field removed.

#### File Operations

**Before:**
```python
file_metadata = client.get_file(experiment_id, file_id)
# file_metadata = {
#     "id": "...",
#     "filename": "...",
#     "path": "...",
#     "checksum": "...",
#     ...
# }
```

**After:**
```python
node_metadata = client.get_file(experiment_id, file_id)
# node_metadata = {
#     "id": "...",         # Node ID
#     "name": "...",       # Filename
#     "pPath": "...",      # Hierarchical path
#     "physicalFile": {    # NEW - nested physical file data
#         "id": "...",
#         "filename": "...",
#         "checksum": "...",
#         ...
#     }
# }
```

**Impact**: File metadata now nested under `physicalFile`. Use `node["physicalFile"]["checksum"]` instead of `file["checksum"]`.

---

### 5. File ID vs Node ID

**Before:**
- File ID referred to the file record
- Used in all file operations

**After:**
- File ID is now **Node ID**
- Node represents the file in the hierarchy
- Physical file ID is separate (in `physicalFile.id`)

**Impact**: Minimal - most code continues to work as `file_id` parameter is now treated as `node_id`.

---

## Migration Guide

### Step 1: Update Client Initialization

Add namespace parameter:

```python
# Get namespace from environment or config
namespace = os.environ.get("ML_DASH_NAMESPACE", "default-namespace")

client = RemoteClient(
    base_url="http://localhost:3000",
    namespace=namespace,
    api_key=token
)
```

### Step 2: Update File Metadata Access

```python
# OLD
filename = file_metadata["filename"]
checksum = file_metadata["checksum"]

# NEW
filename = file_metadata["name"]  # or file_metadata["physicalFile"]["filename"]
checksum = file_metadata["physicalFile"]["checksum"]
```

### Step 3: Use Folder Nodes Instead of Prefixes

```python
# Create folder structure first
folder_node = client.create_folder_node(
    project_id=project_id,
    experiment_id=experiment_id,
    name="models",
    parent_id="ROOT"
)

# Upload file to folder
client.upload_file(
    experiment_id=experiment_id,
    file_path="./model.pt",
    parent_id=folder_node["id"],  # Use folder node ID
    ...
)
```

---

## High-Level API Changes

The high-level `Experiment` class wraps these changes, so impact is minimal:

```python
# This still works with new API
with Experiment(project="my-project", name="exp-1").run as exp:
    exp.files("models").upload("./model.pt")
    exp.files("data").save_json(data, to="config.json")
```

**However**, you now need to pass `namespace` when creating experiments:

```python
from ml_dash import Experiment

# NEW: namespace parameter required
exp = Experiment(
    project="my-project",
    name="exp-1",
    namespace="my-namespace"  # REQUIRED
)
```

---

## Deprecations

The following parameters are deprecated but still accepted (ignored):

1. `prefix` in `upload_file()` - Use `parent_id` instead
2. `prefix` in `list_files()` - Do client-side filtering
3. `experiment_id` in file operations - Not needed with node API (but kept for compatibility)

---

## GraphQL Requirement

The client now requires GraphQL for certain operations:

1. **File listing** - Uses GraphQL instead of REST
2. **ID resolution** - Project slug → ID resolution
3. **Node ID lookup** - Experiment ID → Node ID

**Impact**: GraphQL endpoint must be accessible at `{base_url}/graphql`.

---

## Testing Your Migration

### 1. Test Client Initialization
```python
client = RemoteClient(
    base_url="http://localhost:3000",
    namespace="test-namespace"
)
assert client.namespace == "test-namespace"
```

### 2. Test Experiment Creation
```python
result = client.create_or_update_experiment(
    project="test-project",
    name="test-exp"
)
assert "node" in result
assert "experiment" in result
```

### 3. Test File Operations
```python
# Upload
result = client.upload_file(...)
assert "node" in result
assert "physicalFile" in result

# List
files = client.list_files(experiment_id="...")
assert isinstance(files, list)

# Download
path = client.download_file(experiment_id="...", file_id="node-id")
assert os.path.exists(path)
```

---

## Rollback Plan

If you need to rollback:

1. Use the previous version of ml-dash client
2. Ensure server still has deprecated endpoints enabled (coexistence period)
3. Plan migration during server's deprecation window

---

## Support

For questions or issues:
- Check `PYTHON_CLIENT_MIGRATION.md` for technical details
- Check `MIGRATION_ANALYSIS.md` for full analysis
- File issues on GitHub

---

## Timeline

- **Now**: Breaking changes implemented
- **Next Release**: Deploy with breaking changes
- **Server Deprecation**: Old endpoints will be removed 4-6 weeks after release

---

**Important**: Update your code before the server removes deprecated endpoints!
