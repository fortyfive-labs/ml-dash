# ML-Logger Special Files Summary

## Overview

ML-Logger uses **special `.jsonl` files** that get automatic handling by the sync daemon. This document summarizes which files are special and how they're processed.

---

## ⚡ Special Files (Auto-Synced)

These files have special handling by the daemon:

| File | Purpose | Format | Sync Priority | Endpoint |
|------|---------|--------|---------------|----------|
| `parameters.jsonl` | Experiment parameters | JSONL | High (immediate) | GraphQL `saveParameters` |
| `metrics.jsonl` | All metrics (time-series) | JSONL | Medium (batched, 10s) | REST `/api/v1/metrics/:id/batch` |
| `logs.jsonl` | Structured text logs | JSONL | Low (60s) | (Optional) |

---

## 📄 File Formats

### 1. `parameters.jsonl` ⚡

**Purpose**: Track parameter changes over time

**Format**:
```jsonl
{"timestamp": 1634567890123, "operation": "set", "data": {"learning_rate": 0.001, "batch_size": 32}}
{"timestamp": 1634567891000, "operation": "extend", "data": {"train": {"epochs": 100}}}
{"timestamp": 1634567892000, "operation": "update", "key": "learning_rate", "value": 0.0005}
```

**Operations**:
- `set`: Set/overwrite parameters
- `extend`: Merge with existing parameters (deep merge)
- `update`: Update single key

**Python API**:
```python
# Writes to parameters.jsonl
logger.params.set(learning_rate=0.001, batch_size=32)
logger.params.extend(train=dict(epochs=100))
logger.params.update("learning_rate", 0.0005)
```

**Daemon Behavior**:
1. Reads new lines since last sync
2. Replays operations to compute final state
3. Sends final state to server via GraphQL

**Why append-only?**:
- ✅ Full history of parameter changes
- ✅ Can debug when parameters changed
- ✅ Fast writes (no rewriting)
- ✅ Easy to sync (line number tracking)

---

### 2. `metrics.jsonl` ⚡

**Purpose**: All metrics for the experiment (single file)

**Format**:
```jsonl
{"timestamp": 1634567890123, "step": 0, "metrics": {"train.loss": 0.523, "train.accuracy": 0.85}}
{"timestamp": 1634567891000, "step": 1, "metrics": {"train.loss": 0.501, "train.accuracy": 0.87}}
{"timestamp": 1634567892000, "step": 2, "metrics": {"train.loss": 0.487, "train.accuracy": 0.89}}
{"timestamp": 1634567893000, "step": 0, "metrics": {"val.loss": 0.550, "val.accuracy": 0.83}}
```

**Schema**:
```typescript
{
  timestamp: number,  // Unix timestamp in milliseconds
  step: number,       // Training step/epoch
  metrics: {
    [name: string]: number  // Metric name -> value
  }
}
```

**Python API**:
```python
# Writes to metrics.jsonl
logger.metrics.log(step=0, train_loss=0.523, train_accuracy=0.85)

# With namespace (creates "train.loss", "train.accuracy")
logger.metrics("train").log(step=0, loss=0.523, accuracy=0.85)
```

**Daemon Behavior**:
1. Reads 100 lines (or 10 seconds worth)
2. Converts to batch format:
   ```python
   {
     "train.loss": {
       "times": [1634567890123, 1634567891000, ...],
       "values": [0.523, 0.501, ...]
     },
     "train.accuracy": {
       "times": [1634567890123, 1634567891000, ...],
       "values": [0.85, 0.87, ...]
     }
   }
   ```
3. Sends single POST to `/api/v1/metrics/:experimentId/batch`

**Why single file?**:
- ✅ All metrics in one place
- ✅ Simpler daemon logic (one file to watch)
- ✅ Atomic writes (all metrics for a step together)
- ✅ Efficient batching (one read, not per-metric)

---

### 3. `logs.jsonl` ⚡

**Purpose**: Structured application logs

**Format**:
```jsonl
{"timestamp": 1634567890123, "level": "INFO", "message": "Starting training", "context": {}}
{"timestamp": 1634567891000, "level": "DEBUG", "message": "Loading model", "context": {"model": "resnet50"}}
{"timestamp": 1634567892000, "level": "ERROR", "message": "GPU OOM", "context": {"gpu_id": 0, "allocated": "8GB"}}
```

**Schema**:
```typescript
{
  timestamp: number,
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR",
  message: string,
  context: { [key: string]: any }  // Optional structured data
}
```

**Python API**:
```python
# Writes to logs.jsonl
logger.info("Starting training")
logger.debug("Loading model", model="resnet50")
logger.error("GPU OOM", gpu_id=0, allocated="8GB")
```

**Daemon Behavior**:
- Low priority (not critical for training)
- Batched sync (every 60 seconds)
- Can skip if server doesn't support logs endpoint

---

## 🗂️ Complete File Structure

```
.ml-dash/alice/project-name/experiments/resnet-baseline/
├── parameters.jsonl            # ⚡ Special: Auto-synced to GraphQL
├── metrics.jsonl               # ⚡ Special: Auto-synced to REST batch endpoint
├── logs.jsonl                  # ⚡ Special: Auto-synced (low priority)
├── files/                      # Regular files: Uploaded as multipart
│   ├── checkpoints/
│   │   └── epoch_10.pt
│   ├── images/
│   │   └── confusion_matrix.png
│   └── videos/
│       └── training.mp4
└── .ml-dash.meta.json        # Sync state (line numbers, upload status)
```

---

## 🔄 Daemon Processing Logic

### Startup
1. Load `.ml-dash.meta.json` to get sync state
2. Watch for changes to special files
3. Start periodic sync timer

### On File Change
```python
def on_file_change(filepath):
    if filepath.endswith('.jsonl'):
        # Special file - queue for sync
        if 'parameters' in filepath:
            schedule_sync('parameters', priority='high')
        elif 'metrics' in filepath:
            schedule_sync('metrics', priority='medium')
        elif 'logs' in filepath:
            schedule_sync('logs', priority='low')
    else:
        # Regular file - queue for upload
        schedule_upload(filepath, priority='low')
```

### Periodic Sync (Every 10 seconds)
```python
def periodic_sync():
    # 1. Sync parameters if changed
    if has_new_lines('parameters.jsonl'):
        sync_parameters()

    # 2. Sync metrics (batch)
    if has_new_lines('metrics.jsonl', min_lines=100) or time_since_last_sync > 10:
        sync_metrics_batch()

    # 3. Sync logs (low priority)
    if time_since_last_sync('logs') > 60:
        sync_logs()

    # 4. Upload pending files (when idle)
    if bandwidth_available():
        upload_next_file()
```

---

## 📊 Comparison: Special vs Regular Files

| Aspect | Special `.jsonl` | Regular Files |
|--------|-----------------|---------------|
| **Format** | Text (JSON lines) | Binary (any) |
| **Append** | Yes (line-by-line) | No (full file) |
| **Tracking** | Line number | File hash |
| **Syncing** | Incremental (new lines) | Full file upload |
| **Priority** | High/Medium | Low |
| **Batching** | Yes (100s of lines) | N/A |
| **Endpoint** | REST/GraphQL | Multipart upload |
| **Human Readable** | Yes (can inspect) | No |
| **Version Control** | Friendly | Not friendly |
| **Offline** | Works perfectly | Works (queued) |
| **Resumable** | Yes (line tracking) | Yes (chunked upload) |

---

## 🎯 Benefits of This Design

### 1. **Performance**
- ✅ **Zero network latency** for training loops (just append to file)
- ✅ **Batching** reduces requests by 100x (100 metrics → 1 request)
- ✅ **Async** - daemon handles uploads in background

### 2. **Reliability**
- ✅ **Local-first** - works offline, syncs when online
- ✅ **Resumable** - can restart daemon and continue
- ✅ **Atomic** - line-by-line writes, no corruption

### 3. **Debuggability**
- ✅ **Human readable** - can inspect files with text editor
- ✅ **Replayable** - can re-run from local files
- ✅ **History** - full log of all operations

### 4. **Simplicity**
- ✅ **Three files** - parameters, metrics, logs
- ✅ **JSONL format** - standard, easy to parse
- ✅ **One daemon** - handles all syncing

---

## 🔍 Example: Complete Training Run

### During Training

**Python code**:
```python
logger = ML_Dash(
    namespace="alice",
    workspace="project-name",
    prefix="experiments/resnet-baseline"
)

# Parameters
logger.params.set(learning_rate=0.001, batch_size=32)

# Training loop
for epoch in range(100):
    for batch in range(1000):
        loss = train_step()
        logger.metrics.log(step=epoch*1000+batch, train_loss=loss)

    logger.info(f"Completed epoch {epoch}")
    logger.files.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
```

**Files created**:
```
.ml-dash/alice/project-name/experiments/resnet-baseline/
├── parameters.jsonl        # 1 line
├── metrics.jsonl           # 100,000 lines (100 epochs × 1000 batches)
├── logs.jsonl              # 100 lines (one per epoch)
├── files/checkpoints/
│   ├── checkpoint_epoch_0.pt
│   ├── checkpoint_epoch_1.pt
│   └── ...
└── .ml-dash.meta.json    # Sync state
```

### Daemon Activity

**Timeline**:
```
T+0s:   Parameters written → Synced immediately (high priority)
T+10s:  1000 metrics → Batched and synced
T+20s:  1000 metrics → Batched and synced
...
T+60s:  Logs synced (low priority)
T+300s: First checkpoint uploaded (when idle)
```

**Network requests**:
- 1 GraphQL mutation (parameters)
- 100 REST batch requests (metrics, 1000 lines each)
- 1 logs upload (optional)
- 100 multipart uploads (checkpoints)

**Total: ~202 requests for 100,000+ operations** (500x reduction!)

---

## 🚀 Implementation Priority

### Phase 1: Core Special Files (MVP)
1. ✅ `parameters.jsonl` - Set/extend/update tracking
2. ✅ `metrics.jsonl` - All metrics in one file
3. ✅ `.ml-dash.meta.json` - Sync state
4. ✅ Daemon - Basic sync logic

### Phase 2: Enhanced Features
1. ✅ `logs.jsonl` - Structured logging
2. ✅ File uploads - Multipart binary files
3. ✅ Retry logic - Exponential backoff
4. ✅ Compression - Gzip old metrics

### Phase 3: Advanced
1. ⚠️  RPC operations - makeVideo, glob, etc.
2. ⚠️  Chunked uploads - Large files (>100MB)
3. ⚠️  Delta sync - Only changed parameters
4. ⚠️  Real-time streaming - WebSocket metrics

---

## 📝 Key Takeaways

1. **Three special `.jsonl` files**: `parameters.jsonl`, `metrics.jsonl`, `logs.jsonl`
2. **Append-only**: Never rewrite, only append (fast & safe)
3. **Line tracking**: Daemon tracks line numbers for incremental sync
4. **Batching**: 100s of lines in one request (efficient)
5. **Priority**: Parameters (high), metrics (medium), logs (low), files (low)
6. **Local-first**: Everything works offline, syncs when online
7. **Human-readable**: Can inspect with text editor, grep, etc.
8. **Simple daemon**: Watches files, reads new lines, uploads in background

This design gives you **Netflix-level performance** (local-first) with **Dropbox-level reliability** (sync daemon) while keeping **git-level simplicity** (text files)!
