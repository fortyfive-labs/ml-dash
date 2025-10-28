# ML-Logger Minimal API

**This is the canonical minimal API specification for ML-Logger.**

All features, examples, and tests should align with this specification. Features not listed here should be removed.

---

## Core API Files

### Primary Documentation
1. **`docs/MINIMAL_API.md`** (this file) - Canonical minimal API specification
2. **`docs/quick_start.md`** - User-facing quick start guide

### Deprecated/To Remove
- ❌ `docs/RUN_API_DESIGN.md` - Outdated (refers to "Run" instead of "Experiment")
- ❌ `docs/API_NAMING_DECISION.md` - Historical, no longer relevant
- ❌ `docs/API_CHANGES_SUMMARY.md` - Historical changelog, superseded
- ❌ `RUN_API_IMPLEMENTATION.md` - Root-level implementation doc (outdated)
- ❌ `EXPERIMENT_API_SUMMARY.md` - Root-level summary (outdated)

---

## Minimal API: Core Classes

### 1. Experiment

The main API object representing a single training execution.

```python
from ml_dash import Experiment

experiment = Experiment(
    namespace: str,  # Required: User/team namespace
workspace: str,  # Required: Project workspace
prefix: str,  # Required: Experiment path
remote: str | None = None,  # Optional: Remote server URL
local_root: str = ".ml-logger",  # Local storage directory
readme: str | None = None,  # Searchable description
experiment_id: str | None = None  # Server-side experiment ID
)
```

**Metadata:**
- `experiment.readme` - Markdown description (searchable)
- `experiment.charts` - Programmatic visualization specs (dict)

**Components:**
- `experiment.params` - ParameterManager
- `experiment.metrics` - MetricsLogger
- `experiment.files` - FileManager
- `experiment.logs` - LogManager

**Lifecycle:**
- `experiment.run()` - Mark as started (3 patterns: direct, context manager, decorator)
- `experiment.complete()` - Mark as completed
- `experiment.fail(error)` - Mark as failed

**Logging shortcuts:**
- `experiment.logs.info(message, **context)` - Log info message
- `experiment.logs.error(message, **context)` - Log error message
- `experiment.logs.warning(message, **context)` - Log warning message

---

### 2. Autolog (Auto-configured Experiment)

```python
from ml_dash.autolog import experiment

# Auto-configured from environment variables:
# - ML_LOGGER_NAMESPACE
# - ML_LOGGER_WORKSPACE
# - ML_LOGGER_PREFIX
# - ML_LOGGER_REMOTE

experiment.params.set(learning_rate=0.001)
experiment.metrics.log(step=0, loss=0.5)
```

---

## Minimal API: Components

### ParameterManager (`experiment.params`)

These parameters should be saved as dot-separated keys (e.g. `model.layers`)
on the server.

**Methods:**
- `params.set(**kwargs)` - Set parameters (replaces existing)
- `params.extend(**kwargs)` - Deep merge with existing parameters
- `params.update(key, value)` - Update single parameter

**File:** `parameters.jsonl` (append-only JSONL)

**Example:**
```python
experiment.params.set(
    learning_rate=0.001,
    batch_size=32,
    model=dict(
        name="resnet50",
        layers=50
    )
)
```

---

### MetricsLogger (`experiment.metrics`)

**Methods:**
- `metrics.log(step, **metrics)` - Log metrics immediately
- `metrics.collect(**metrics)` - Collect metrics for aggregation
- `metrics.flush(_aggregation="mean", step, **additional)` - Flush collected metrics
- `metrics(namespace)` - Create namespaced logger

**File:** `metrics.jsonl` (append-only JSONL)

**Example:**
```python
# Immediate logging
experiment.metrics.log(step=0, train_loss=0.5, val_loss=0.6)

# Namespaced
train_metrics = experiment.metrics("train")
train_metrics.log(step=0, loss=0.5)

# Aggregation
for batch in batches:
    experiment.metrics.collect(loss=batch_loss)
    
experiment.metrics.flush(_aggregation="mean", step=epoch)
```

---

### FileManager (`experiment.files`)

**Methods:**
- `files.save(data, filename)` - Save any data (JSON, pickle, torch, numpy)
- `files.save_pkl(data, filename)` - Save pickle file
- `files(namespace)` - Create namespaced file manager

**Directory:** `files/` subdirectory

**Example:**
```python
# Save checkpoint
experiment.files.save(model.state_dict(), "checkpoint.pt")

# Save JSON
experiment.files.save({"results": metrics}, "results.json")

# Namespaced
checkpoints = experiment.files("checkpoints")
checkpoints.save(model.state_dict(), "model_epoch_10.pt")
# Saves to: files/checkpoints/model_epoch_10.pt
```

**NOT in minimal API:**
- ❌ `save_image()` - Image logging (remove)
- ❌ `save_video()` - Video logging (remove)
- ❌ `make_video()` - Video conversion (remove)
- ❌ Image/video helper methods

---

### LogManager (`experiment.logs`)

**Methods:**
- `logs.info(message, **context)` - Log info message
- `logs.error(message, **context)` - Log error message
- `logs.warning(message, **context)` - Log warning message
- `logs.debug(message, **context)` - Log debug message

**File:** `logs.jsonl` (append-only JSONL)

**Example:**
```python
experiment.logs.info("Training started", learning_rate=0.001)
experiment.logs.error("Training failed", error=str(e))

# Shortcut via experiment
experiment.info("Epoch completed", epoch=10, loss=0.5)
```

---

## Minimal API: Lifecycle Patterns

### Pattern 1: Direct Call
```python
experiment.run()  # Mark as started

for epoch in range(100):
    loss = train_epoch()
    experiment.metrics.log(step=epoch, loss=loss)

experiment.complete()  # Mark as completed
```

### Pattern 2: Context Manager
```python
with experiment.run():
    experiment.params.set(learning_rate=0.001)

    for epoch in range(100):
        loss = train_epoch()
        experiment.metrics.log(step=epoch, loss=loss)

    # Auto-completes on success, auto-fails on exception
```

### Pattern 3: Decorator
```python
@experiment.run
def train(config):
    experiment.params.set(**config)

    for epoch in range(100):
        loss = train_epoch()
        experiment.metrics.log(step=epoch, loss=loss)

    return final_metrics

result = train(config)  # Auto-completes on success
```

---

## File Structure

```
.ml-logger/{namespace}/{workspace}/{prefix}/
├── parameters.jsonl         # Parameter operations
├── metrics.jsonl            # All metrics
├── logs.jsonl               # Text logs
├── files/                   # Saved files and artifacts
│   ├── checkpoints/         # (if using namespaced files)
│   └── ...
└── .ml-logger.meta.json     # Experiment metadata
```

**Metadata file (`.ml-logger.meta.json`):**
```json
{
  "namespace": "alice",
  "workspace": "project",
  "prefix": "exp1/trial-001",
  "remote": null,
  "experiment_id": null,
  "readme": "Experiment description...",
  "charts": {},
  "status": "completed",
  "started_at": 1760326054.033515,
  "completed_at": 1760326054.0341089,
  "hostname": "host.local",
  "updated_at": 1760326054.03411
}
```

---

## What's NOT in the Minimal API

### Features to Remove:
1. ❌ **Image logging** - `save_image()`, image helpers
2. ❌ **Video logging** - `save_video()`, `make_video()`, video conversion
3. ❌ **Authentication** - `ml-logger login`, token management
4. ❌ **`.logrc` configuration** - YAML config file support
5. ❌ **Tags** - `experiment.tags` (use readme for search instead)
6. ❌ **Advanced file operations** - Complex file queries/filters
7. ❌ **Remote sync daemon** - Background sync (not implemented yet)
8. ❌ **Old Logger API** - The original `Logger` class

### Examples to Remove:
1. ❌ Examples with image logging
2. ❌ Examples with video logging
3. ❌ Examples with authentication
4. ❌ Examples with tags
5. ❌ Examples with `.logrc` config

### Documentation to Remove/Deprecate:
1. ❌ `docs/RUN_API_DESIGN.md` - Refers to old "Run" API
2. ❌ `docs/API_NAMING_DECISION.md` - Historical decision doc
3. ❌ `docs/API_CHANGES_SUMMARY.md` - Historical changes
4. ❌ `RUN_API_IMPLEMENTATION.md` - Root-level outdated doc
5. ❌ `EXPERIMENT_API_SUMMARY.md` - Root-level outdated doc

---

## Minimal Example

```python
from ml_dash import Experiment

# 1. Create experiment
experiment = Experiment(
    namespace="alice",
    workspace="image-classification",
    prefix="exp1/trial-001",
    readme="Testing ResNet50 with learning_rate=0.001"
)

# 2. Set parameters
experiment.params.set(
    model="resnet50",
    learning_rate=0.001,
    batch_size=32
)

# 3. Training with lifecycle management
with experiment.run():
    # Training loop
    for epoch in range(100):
        train_loss = train_epoch()

        # Log metrics
        experiment.metrics.log(
            step=epoch,
            train_loss=train_loss
        )

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            experiment.files.save(
                model.state_dict(),
                f"checkpoint_{epoch}.pt"
            )

        # Log progress
        experiment.info(f"Epoch {epoch} completed", loss=train_loss)

    # Save final results
    experiment.files.save(
        {"final_loss": train_loss},
        "results.json"
    )

# Experiment is automatically marked as completed

print(f"✓ Experiment completed: {experiment.local_path}")
```

---

## Configuration

### Environment Variables (for autolog)
```bash
export ML_LOGGER_NAMESPACE="your-username"
export ML_LOGGER_WORKSPACE="your-project"
export ML_LOGGER_PREFIX="exp1/trial-001"      # Optional
export ML_LOGGER_REMOTE="http://localhost:3001"  # Optional
```

### Programmatic Configuration
```python
experiment = Experiment(
    namespace="alice",
    workspace="project",
    prefix="exp1/trial-001",
    remote="http://localhost:3001",  # Optional remote server
    local_root=".ml-logger"           # Local storage
)
```

---

## Summary

**Core concepts:**
- One `Experiment` = one training execution
- Four components: `params`, `metrics`, `files`, `logs`
- Three lifecycle patterns: direct call, context manager, decorator
- Local-first: All writes go to `.ml-logger/` directory
- Searchable metadata: `readme`, `charts`

**What's included:**
- ✅ Basic parameter logging
- ✅ Metric logging with aggregation
- ✅ File saving (JSON, pickle, torch, numpy)
- ✅ Text logging
- ✅ Namespaced metrics and files
- ✅ Three lifecycle patterns
- ✅ Auto-configured experiments (autolog)
- ✅ Searchable readme

**What's excluded:**
- ❌ Image/video logging
- ❌ Authentication
- ❌ Tags
- ❌ `.logrc` config files
- ❌ Advanced file operations
- ❌ Remote sync (not implemented)
- ❌ Old Logger API
