# Architecture & Design

This document provides an in-depth look at DreamLake's architecture, design decisions, and internal workings.

## Overview

DreamLake is built with a clean, modular architecture that supports both local filesystem and remote API server backends. The design emphasizes simplicity, flexibility, and ease of use while maintaining powerful functionality for ML experiment metricing.

### High-Level Architecture

```{mermaid}
flowchart TB
    User[User Code] --> Experiment[Experiment Manager]
    Experiment --> Builder[Builder APIs]
    Builder --> Log[LogBuilder]
    Builder --> Params[ParametersBuilder]
    Builder --> Metric[MetricBuilder]
    Builder --> Files[FileBuilder]

    Log --> Backend{Backend Layer}
    Params --> Backend
    Metric --> Backend
    Files --> Backend

    Backend --> Local[LocalStorage]
    Backend --> Remote[RemoteClient]

    Local --> FS[Filesystem<br/>JSON/JSONL]
    Remote --> API[REST API]

    API --> MongoDB[(MongoDB)]
    API --> S3[(S3/MinIO)]

    style Experiment fill:#e1f5ff
    style Backend fill:#fff4e1
    style Local fill:#e8f5e9
    style Remote fill:#f3e5f5
```

## Core Components

### 1. Experiment Manager

The `Experiment` class is the entry point for all DreamLake operations. It:

- **Manages lifecycle**: Creation, opening, closing of experiment experiments
- **Handles backends**: Automatically selects LocalStorage or RemoteClient based on configuration
- **Provides builder access**: Returns builder instances for logs, parameters, metrics, and files
- **Supports multiple patterns**: Context manager, decorator, or direct instantiation

**Key responsibilities**:
```text
Experiment
├── Lifecycle management (open/close)
├── Backend initialization (local or remote)
├── Builder factory methods
├── Experiment metadata management
└── Error handling and recovery
```

### 2. Builder APIs

DreamLake uses the **Builder Pattern** to provide a fluent, chainable API for data operations:

#### LogBuilder
```python
experiment.log("Message", level="info", metadata={...})
```
- Structured logging with 5 levels (debug, info, warn, error, fatal)
- Automatic timestamping and sequence numbering
- Metadata support for structured data

#### ParametersBuilder
```python
experiment.parameters().set(lr=0.001, batch_size=32)
```
- Stores hyperparameters and configuration
- **Automatic flattening**: Nested dicts converted to dot notation
  - Input: `{"model": {"layers": 50}}`
  - Stored: `{"model.layers": 50}`
- Upsert behavior: Updates existing values

#### MetricBuilder
```python
experiment.metric("loss").append(value=0.5, epoch=0)
experiment.metric("loss").append_batch([...])
```
- Time-series metrics metricing
- Flexible schema (any fields)
- Efficient batch operations
- Read with pagination support

#### FileBuilder
```python
experiment.file(file_prefix="model.pth", prefix="/models").save()
```
- File upload and organization
- Checksum validation (SHA256)
- Metadata and tagging
- Hierarchical organization with prefixes

### 3. Backend Layer

The backend layer abstracts storage implementation, allowing DreamLake to work with different backends without changing user code.

#### LocalStorage

**Filesystem Structure**:
```
<root_path>/
└── <project>/
    └── <experiment>/
        ├── experiment.json          # Experiment metadata
        ├── parameters.json       # Hyperparameters
        ├── logs/
        │   └── logs.jsonl       # Log entries (JSON Lines)
        ├── metrics/
        │   └── <metric_name>/
        │       ├── metadata.json
        │       └── data.jsonl   # Time-series data
        └── files/
            ├── .files_metadata.json
            └── <file_id>/
                └── <filename>
```

**Data Formats**:
- **JSON**: Structured metadata (experiment.json, parameters.json)
- **JSONL** (JSON Lines): Append-only logs and metrics
- **Raw Files**: Binary files stored with original names

**Advantages**:
- ✅ No server required
- ✅ Easy to inspect and debug
- ✅ Fast for local development
- ✅ Works offline
- ✅ Git-friendly for small experiments

#### RemoteClient

**REST API Communication**:
```
POST   /projects/{project}/experiments
POST   /experiments/{id}/logs
POST   /experiments/{id}/parameters
POST   /experiments/{id}/metrics/{name}
POST   /experiments/{id}/metrics/{name}/batch
POST   /experiments/{id}/files
GET    /experiments/{id}/metrics/{name}
GET    /experiments/{id}/metrics
GET    /experiments/{id}/files
```

**Authentication**:
- JWT tokens via `Authorization: Bearer <token>` header
- Auto-generation from `user_name` parameter (development mode)
- Custom API key support

**Advantages**:
- ✅ Centralized storage and sharing
- ✅ Team collaboration
- ✅ Scalable for large experiments
- ✅ Query and search capabilities
- ✅ Web UI integration

## Data Flow

### Local Mode Flow

```{mermaid}
sequenceDiagram
    participant User
    participant Experiment
    participant LocalStorage
    participant FS as Filesystem

    User->>Experiment: Create experiment
    Experiment->>LocalStorage: Initialize storage
    LocalStorage->>FS: Create directories

    User->>Experiment: log("message")
    Experiment->>LocalStorage: write_log(...)
    LocalStorage->>FS: Append to logs.jsonl

    User->>Experiment: parameters().set(...)
    Experiment->>LocalStorage: write_parameters(...)
    LocalStorage->>FS: Write parameters.json

    User->>Experiment: close()
    Experiment->>LocalStorage: finalize()
```

### Remote Mode Flow

```{mermaid}
sequenceDiagram
    participant User
    participant Experiment
    participant RemoteClient
    participant API as REST API
    participant DB as MongoDB

    User->>Experiment: Create experiment
    Experiment->>RemoteClient: create_experiment(...)
    RemoteClient->>API: POST /projects/{ws}/experiments
    API->>DB: Insert experiment doc
    DB-->>API: Experiment ID
    API-->>RemoteClient: Experiment data

    User->>Experiment: log("message")
    Experiment->>RemoteClient: create_log_entries([...])
    RemoteClient->>API: POST /experiments/{id}/logs
    API->>DB: Insert log docs

    User->>Experiment: close()
    Experiment->>RemoteClient: (finalize if needed)
```

## Design Decisions

### 1. Builder Pattern

**Why?**
- **Fluent API**: Chainable methods for better readability
- **Lazy initialization**: Builders created on-demand
- **Separation of concerns**: Each builder focuses on one data type
- **Extensibility**: Easy to add new data types

**Example**:
```python
# Clean, readable API
experiment.metric("accuracy").append(value=0.95, epoch=10)

# vs procedural approach
experiment.append_metric("accuracy", {"value": 0.95, "epoch": 10})
```

### 2. Upsert Behavior

**What**: Experiments can be reopened and updated

**Why?**
- **Recovery**: Resume after crashes or interruptions
- **Iterative development**: Add data to existing experiments
- **Flexibility**: Update metadata, add new metrics/logs

**Implementation**:
- Local: Check if experiment directory exists, merge data
- Remote: API checks experiment existence, merges on server

### 3. Auto-Creation

**What**: Automatically creates namespace → project → folder hierarchy

**Why?**
- **Less boilerplate**: No manual directory/project creation
- **Better UX**: Focus on experiment, not setup
- **Convention over configuration**: Sensible defaults

**Example**:
```python
# This automatically creates:
# - Namespace (if remote)
# - Project "my-project"
# - Folder "/experiments/2024"
# - Experiment "baseline"
Experiment(
    name="baseline",
    project="my-project",
    folder="/experiments/2024",
    local_prefix=".ml-dash",
        local_path=".ml-dash"
)
```

### 4. Dual Backend Support

**Why?**
- **Flexibility**: Local for development, remote for production
- **Gradual adoption**: Start local, migrate to remote when ready
- **Offline capability**: Work without network access
- **Testing**: Easy to test with local mode

**Trade-offs**:

| Aspect | Local Mode | Remote Mode |
|--------|-----------|-------------|
| Setup | ✅ Zero setup | ⚠️ Requires server |
| Performance | ✅ Fast writes | ⚠️ Network latency |
| Collaboration | ❌ File sharing only | ✅ Built-in sharing |
| Querying | ❌ Manual file inspection | ✅ API queries |
| Scalability | ⚠️ Limited by disk | ✅ Scales horizontally |

### 5. JSON/JSONL Format

**Why?**
- **Human-readable**: Easy to inspect and debug
- **Language-agnostic**: Any tool can read
- **Append-friendly**: JSONL for logs/metrics
- **Git-friendly**: Text-based diffs

**JSONL (JSON Lines)** for append operations:
```json
{"timestamp": "2024-01-01T00:00:00Z", "message": "Log 1"}
{"timestamp": "2024-01-01T00:00:01Z", "message": "Log 2"}
{"timestamp": "2024-01-01T00:00:02Z", "message": "Log 3"}
```

Benefits:
- ✅ No need to load entire file
- ✅ Append-only (no file rewrites)
- ✅ Stream-friendly
- ✅ Fault-tolerant (partial reads work)

## Extensibility

### Custom Storage Backends

DreamLake's architecture allows for custom storage backends by implementing the storage interface:

```python
class CustomStorage:
    def create_experiment(self, name, project, **kwargs):
        # Create experiment
        pass

    def write_log(self, experiment_id, log_entry):
        # Store log
        pass

    def write_parameters(self, experiment_id, params):
        # Store parameters
        pass

    # ... other methods
```

### Future Extensibility Points

**Planned**:
- Plugin system for custom data types
- Middleware for data transformation
- Custom serialization formats
- Storage adapters (PostgreSQL, DynamoDB, etc.)
- Event hooks (pre-save, post-save)

## Performance Considerations

### Batch Operations

For high-throughput scenarios, use batch operations:

```python
# Instead of multiple appends
for data in dataset:
    experiment.metric("metric").append(**data)  # ❌ Slow

# Use batch append
batch_data = [{"value": x, "step": i} for i, x in enumerate(values)]
experiment.metric("metric").append_batch(batch_data)  # ✅ Fast
```

**Performance gains**:
- Local: 10-50x faster (reduces filesystem operations)
- Remote: 20-100x faster (reduces network requests)

### File Upload Optimization

- **Chunked upload**: For large files (remote mode)
- **Checksum validation**: Ensures data integrity
- **Concurrent uploads**: Multiple files in parallel (planned)

### Caching Strategy

**Current**:
- Experiment metadata cached in memory
- Parameters cached until update
- No caching for logs/metrics (append-only)

**Future**:
- Configurable write batching
- Read caching for metrics
- Lazy loading for large datasets

## Security

### Authentication (Remote Mode)

**JWT Tokens**:
- Standard Bearer token authentication
- Configurable expiration (default: 30 days)
- Secret key must match server configuration

**Development Mode**:
```python
# Auto-generates JWT from username
Experiment(remote="...", user_name="alice")
# Equivalent to providing a JWT token
```

**Production Mode**:
```python
# Use proper API key from authentication service
Experiment(remote="...", api_key="actual-jwt-token")
```

### Data Security

**In Transit**:
- Use HTTPS for remote connections
- TLS for MongoDB connections

**At Rest**:
- Local: Standard filesystem permissions
- Remote: Encryption at database level
- S3: Server-side encryption

## Comparison with Other Tools

| Feature | DreamLake | MLflow | Weights & Biases | Neptune.ai |
|---------|-----------|---------|------------------|------------|
| **Local Mode** | ✅ First-class | ✅ Yes | ❌ Cloud-only | ❌ Cloud-only |
| **Self-hosted** | ✅ Easy | ✅ Yes | ❌ Enterprise only | ❌ No |
| **Offline Work** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **File Storage** | ✅ Built-in | ✅ Artifact store | ✅ Yes | ✅ Yes |
| **Learning Curve** | ✅ Low | ⚠️ Medium | ⚠️ Medium | ⚠️ Medium |
| **Setup Time** | ✅ < 1 min | ⚠️ 5-10 min | ✅ 2 min | ✅ 2 min |

**DreamLake's sweet spot**:
- Quick local experiments with zero setup
- Easy transition to collaborative remote mode
- Simple, intuitive API
- Full control over your data

## Future Roadmap

### Short Term (v0.3)
- [ ] Hybrid mode (local + remote sync)
- [ ] Query API for searching experiments
- [ ] Web UI for visualization
- [ ] Batch file uploads

### Medium Term (v0.4-0.5)
- [ ] Real-time streaming API
- [ ] Experiment comparison tools
- [ ] Plugin system
- [ ] Integration with popular frameworks

### Long Term (v1.0+)
- [ ] Distributed training support
- [ ] Advanced query language
- [ ] Multi-cloud support
- [ ] Enterprise features (RBAC, audit logs)

## See Also

- [Getting Started](getting-started.md) - Quick start guide
- [Local vs Remote](local-vs-remote.md) - Choosing the right mode
- [Deployment Guide](deployment.md) - Setting up your own server
- [API Reference](api/modules.rst) - Detailed API documentation
