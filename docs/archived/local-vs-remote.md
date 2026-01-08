# Local vs Remote Mode

ML-Dash operates in two modes: **Local** (filesystem) and **Remote** (API + Cloud storage). Understanding the differences helps you choose the right mode for your use case.

## Local Mode

Local mode stores all data on your local filesystem in a `.dash/` directory.

### When to Use Local Mode

- **Development**: Rapid prototyping and testing
- **Single machine**: Running on your local laptop/workstation
- **Offline work**: No internet connection required
- **Quick experiments**: Simple experiments that don't need cloud storage
- **Privacy**: Keep all data local

### Creating a Local Experiment

```python
from ml_dash import Experiment

exp = Experiment(
    prefix="owner/my-project/my-experiment",
    local_path=".dash/owner/my-project/my-experiment"
)

with exp.run:
    import torch.nn as nn

    net = nn.Sequential(nn.Linear(), ...)

    exp.log("Running in local mode")
    exp.params.set(batch_size=32)
    exp.metrics("train").log(loss=0.5)
    exp.files("models").save(net, "model.pth")
```

### Local Storage Structure

```
.dash/
└── owner/
    └── my-project/
        └── my-experiment/
        ├── logs.jsonl              # Log entries (JSONL format)
        ├── parameters.json         # Parameters (JSON)
        ├── metrics/                 # Time-series data
        │   └── loss/
        │       ├── data.jsonl      # Metric data points
        │       └── metadata.json   # Metric metadata
        └── files/                  # Uploaded files
            └── models/
                └── model.pth
```

### Advantages of Local Mode

- **Fast**: No network latency
- **Simple**: No server setup required
- **Portable**: Copy `.dash/` directory to move experiments
- **No costs**: No cloud storage fees

### Disadvantages of Local Mode

- **No sharing**: Can't easily share with team
- **Single machine**: Tied to one computer
- **No web UI**: Can't browse experiments in browser
- **Limited scale**: Large experiments may fill disk

## Remote Mode

Remote mode stores data in MongoDB (metadata) and S3 (large files), accessed via API.

### When to Use Remote Mode

- **Team collaboration**: Share experiments with team
- **Cloud training**: Training on cloud GPUs
- **Large scale**: Many experiments or large files
- **Web UI**: Browse experiments in web interface
- **Production**: Production ML workflows

### Creating a Remote Experiment

```python
from ml_dash import Experiment

# With username (simpler for development)
exp = Experiment(
    prefix="owner/my-project/my-experiment",
    remote="https://api.dash.ml",     # API endpoint
    user_name="your-username"            # Authentication
)

with exp.run:
    exp.log("Running in remote mode")
    exp.params.set(batch_size=32)
    exp.metrics("train").log(loss=0.5)
    exp.files("model.pth", prefix="/models")

# Or with API key (advanced)
exp = Experiment(
    prefix="owner/my-project/my-experiment",
    remote="https://api.dash.ml",     # API endpoint
    api_key="your-api-key-here"          # Authentication
)

with exp.run:
    exp.log("Running in remote mode")
    exp.params.set(batch_size=32)
    exp.metrics("train").log(loss=0.5)
    exp.files("model.pth", prefix="/models")
```

### Remote Storage Architecture

```
MongoDB:
- Experiment metadata
- Logs (recent)
- Parameters
- Metric metadata
- File metadata

S3:
- Files (models, datasets, etc.)
- Archived logs (old logs moved from MongoDB)
- Metric chunks (old metric data moved from MongoDB)
```

### Advantages of Remote Mode

- **Collaborative**: Share with team members
- **Scalable**: Handle large volumes of data
- **Accessible**: Access from anywhere
- **Durable**: Data backed up in cloud
- **Web UI**: View experiments in browser

### Disadvantages of Remote Mode

- **Requires server**: Need to run API server
- **Network dependency**: Requires internet connection
- **Costs**: S3 storage and MongoDB costs
- **Latency**: Network requests slower than local

## Comparison Table

| Feature | Local Mode | Remote Mode |
|---------|-----------|-------------|
| Setup | None | Requires server |
| Speed | Fast (local disk) | Slower (network) |
| Collaboration | No | Yes |
| Scalability | Limited by disk | Unlimited (S3) |
| Cost | Free | Cloud storage costs |
| Offline work | Yes | No |
| Web UI | No | Yes |
| Data backup | Manual | Automatic |

## Switching Between Modes

You can't directly convert between modes, but you can export/import data.

### Export from Local

```python
# Local data is in .dash/ directory
# Copy the entire directory to back up or share
```

### Start Local, Move to Remote Later

```python
# Development (local)
exp = Experiment(prefix="owner/dev/experiment", local_path=".dash/owner/dev/experiment")

with exp.run:
    # Develop your code...
    pass

# Production (remote)
exp = Experiment(prefix="owner/prod/experiment", remote="https://api", api_key="key")

with exp.run:
    # Run at scale...
    pass
```

## Environment Variables

Set default mode using environment variables:

```bash
# Local mode
export DREAMLAKE_LOCAL_PATH="./experiments"

# Remote mode
export DREAMLAKE_API_URL="https://api.dash.ai"
export DREAMLAKE_API_KEY="your-api-key"
```

Then in code:

```python
import os
from ml_dash import Experiment

# Will use environment variables
exp = Experiment(
    prefix="owner/my-project/experiment",
    local_path=os.getenv("DREAMLAKE_LOCAL_PATH"),
    remote=os.getenv("DREAMLAKE_API_URL"),
    api_key=os.getenv("DREAMLAKE_API_KEY")
)

with exp.run:
    pass
```

## Hybrid Approach

Run locally during development, remote in production:

```python
import os
from ml_dash import Experiment

# Check if running in production
is_production = os.getenv("ENVIRONMENT") == "production"

if is_production:
    # Remote mode for production
    experiment_config = {
        "remote": "https://api.dash.ai",
        "api_key": os.getenv("DREAMLAKE_API_KEY")
    }
else:
    # Local mode for development
    experiment_config = {
        "local_path": "./experiments"
    }

exp = Experiment(prefix="owner/ml/experiment", **experiment_config)

with exp.run:
    exp.log("Starting training")
    # Your training code...
```

## Best Practices

1. **Development**: Start with local mode for fast iteration
2. **Production**: Use remote mode for team collaboration
3. **Backup**: Regularly back up `.dash/` in local mode
4. **Environment vars**: Use environment variables for configuration
5. **Testing**: Test both modes before deploying

## Decision Guide

Choose **Local Mode** if:
- Working alone
- Rapid prototyping
- Small experiments
- No cloud infrastructure
- Privacy concerns

Choose **Remote Mode** if:
- Working with a team
- Large-scale experiments
- Cloud training
- Need web UI
- Production workflows

## See Also

**Deployment & Operations:**
- **[Deployment Guide](deployment.md)** - Deploy your own ML-Dash server (Docker, Kubernetes, Cloud)
- **[Architecture](architecture.md)** - Understand the technical differences between modes
- **[FAQ](faq.md)** - When should I use local vs remote mode?

**Getting Started:**
- [Complete Examples](complete-examples.md) - Full examples for both modes
- [Getting Started](getting-started.md) - Quick start tutorial
