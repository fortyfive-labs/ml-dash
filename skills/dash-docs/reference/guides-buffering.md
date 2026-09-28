# Background Buffering

ML-Dash includes a comprehensive background buffering system that eliminates I/O blocking during training. All write operations (logs, metrics, tracks, files) are automatically batched and executed in background threads.

## Overview

The buffering system provides:
- **Non-blocking writes**: All operations return immediately
- **Automatic batching**: Groups writes for efficiency
- **Parallel uploads**: Uses ThreadPoolExecutor for files
- **Graceful error handling**: Training continues even if uploads fail
- **Progress feedback**: Shows flush status after training

## How It Works

```python
from ml_dash import Experiment

with Experiment("my-project/exp").run as experiment:
    # All of these return immediately (non-blocking)
    experiment.log("Training started")
    experiment.params.set(learning_rate=0.001)

    for epoch in range(100):
        for batch in range(1000):
            # Non-blocking metric writes
            experiment.metrics("train").log(
                epoch=epoch,
                batch=batch,
                loss=0.5
            )

    # Automatic flush on context exit
    # Shows: "[ML-Dash] Flushing buffered data..."
```

## Flush Triggers

Data is automatically flushed when:

1. **Time-based**: Every 5 seconds (default)
2. **Size-based**: When queue reaches 100 items (default)
3. **Manual**: When you call `experiment.flush()`
4. **Context exit**: When the experiment context manager exits

## Configuration

### Environment Variables

```bash
# Enable/disable buffering (default: true)
export ML_DASH_BUFFER_ENABLED=true

# Flush interval in seconds (default: 5.0)
export ML_DASH_FLUSH_INTERVAL=5.0

# Batch sizes (default: 100)
export ML_DASH_LOG_BATCH_SIZE=100
export ML_DASH_METRIC_BATCH_SIZE=100
export ML_DASH_TRACK_BATCH_SIZE=100

# File upload workers (default: 4)
export ML_DASH_FILE_UPLOAD_WORKERS=4

# Retries per batch on network errors, 5xx and 429 (defaults: 3, 1.0 s, 30.0 s)
export ML_DASH_MAX_RETRIES=3
export ML_DASH_RETRY_BASE_DELAY=1.0
export ML_DASH_RETRY_MAX_DELAY=30.0
```

### Programmatic Configuration

```python
from ml_dash import Experiment
from ml_dash.buffer import BufferConfig

# Custom buffer configuration
config = BufferConfig(
    flush_interval=10.0,      # Flush every 10 seconds
    log_batch_size=200,       # Batch 200 logs
    metric_batch_size=500,    # Batch 500 metrics
    file_upload_workers=8     # 8 parallel file uploads
)

with Experiment("my-project/exp", buffer_config=config).run as exp:
    # Use custom configuration
    exp.log("Using custom buffer config")
```

## Manual Flushing

Force an immediate flush at any time:

```python
with Experiment("my-project/exp").run as experiment:
    # Training loop
    for epoch in range(100):
        train_model()
        experiment.metrics("train").log(loss=loss)

    # Force flush before saving checkpoint
    experiment.flush()

    # Now safe to save model knowing all metrics are uploaded
    torch.save(model, "checkpoint.pt")
```

`experiment.flush()` is the call that sends data. `metrics.flush()` and
`metrics("name").flush()` are no-ops kept for chaining
(`exp.metrics.log(epoch=1).flush()`); they send nothing.

If any data cannot be sent, `experiment.flush()` raises `NetworkError` with the counts
not sent. That data stays buffered and is retried on the next flush.

## Disabling Buffering

For debugging or special cases, you can disable buffering:

```python
import os
os.environ["ML_DASH_BUFFER_ENABLED"] = "false"

# Or programmatically
from ml_dash.buffer import BufferConfig

config = BufferConfig(buffer_enabled=False)
with Experiment("my-project/exp", buffer_config=config).run as exp:
    # All writes are immediate (blocking)
    exp.log("Immediate write")
```

## Performance Benefits

### Without Buffering (Blocking)
```python
# Each write blocks for ~100-500ms
for i in range(1000):
    experiment.log(f"Step {i}")  # Blocks!
    # Total time: 1000 * 100ms = 100+ seconds
```

### With Buffering (Non-blocking)
```python
# Writes return immediately
for i in range(1000):
    experiment.log(f"Step {i}")  # Returns instantly!
    # Total time: <1 second
```

**Result**: 10-100x speedup for high-frequency logging!

## Progress Messages

When the experiment completes, you'll see:

```
[ML-Dash] Flushing buffered data...
[ML-Dash]   - 1000 log(s), 100 metric(s), 50 track(s), 10 file(s)
[ML-Dash]   Uploading 10 file(s)...
[ML-Dash]   [1/10] Uploaded frame_0001.jpg
[ML-Dash]   [2/10] Uploaded frame_0002.jpg
...
[ML-Dash] ✓ All data flushed successfully
✓ Experiment completed: my-experiment
```

## Error Handling

Training is not interrupted by upload failures, but failures are not hidden either:

- **During the run**, a batch that fails keeps its place and is retried in the
  background at most once per flush interval (never sooner than a server
  `Retry-After`). Each failure logs a warning.
- **Network errors, 5xx and 429** are retried within a flush up to
  `ML_DASH_MAX_RETRIES` times with exponential backoff. A server `Retry-After`
  (seconds or HTTP date) is the minimum wait. If it exceeds `ML_DASH_RETRY_MAX_DELAY`,
  the flush gives up at once and the batch is not sent before that time.
- **On close** (`exp.run.complete()` or leaving `with exp.run:`), each queue gets one
  more bounded attempt. The attempt stops at the first batch that still fails. If anything was not sent, or a file
  upload failed, close raises `NetworkError` with the counts. A run that was
  completing is then marked **FAILED**, not COMPLETED. An exception raised by
  your own code in the `with` block is kept and re-raised. The close failure is logged
  (logger `ml_dash.run`) and, on Python 3.11+, added as a note on your exception.
- Close waits for an in-progress background flush without a timeout. After that,
  the time taken is bounded by the retry settings, the HTTP timeouts and upload sizes, not by a fixed deadline.

```python
with Experiment("my-project/exp").run as experiment:
    # Even if network fails, training continues
    for i in range(10000):
        experiment.log(f"Step {i}")
        # If upload fails, warning is shown but training continues

# Data that could not be sent is reported here, as NetworkError
```

Values that are NaN or infinite cannot be sent to the server. In remote mode,
`metrics(...).log()` rejects them at once with `ValueError`. To aggregate
values that may be NaN, use `buffer()`/`summary_cache.store()`; summaries leave NaN out.

## Thread Safety

The buffering system is fully thread-safe:

```python
from concurrent.futures import ThreadPoolExecutor

with Experiment("my-project/exp").run as experiment:
    def worker(i):
        experiment.log(f"Worker {i}")
        experiment.metrics("train").log(step=i, loss=0.5)

    # Safe to use from multiple threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(worker, range(1000))
```

## Best Practices

1. **Let it auto-flush**: Don't call `flush()` unless you need guarantees before checkpoints
2. **Use context managers**: Ensures all data is flushed on exit
3. **Monitor progress**: Watch the flush messages to understand batching behavior
4. **Adjust batch sizes**: Increase for very high-frequency logging
5. **Keep buffering enabled**: It's designed for production use

## Technical Details

### Architecture

- **Single background thread**: One daemon thread per experiment
- **Resource-specific queues**: Separate queues for logs, metrics, tracks, files
- **No timeout on close**: Waits indefinitely for all data to flush
- **Temp file cleanup**: Automatically cleans up temporary files after upload

### File Handling

When you save files via methods like `save_image()`, `save_json()`, etc.:

1. Content is written to a temporary file
2. File upload is queued in the buffer
3. Temporary file is kept until upload completes
4. Buffer manager cleans up temp file after successful upload
5. All cleanup happens automatically in the background

This ensures:
- No file handle leaks
- No disk space waste
- Proper cleanup even if uploads are delayed

## Troubleshooting

### Data not appearing immediately

This is expected! Data is buffered and flushed periodically. If you need immediate visibility:

```python
experiment.log("Important checkpoint")
experiment.flush()  # Force immediate upload
```

### Memory usage growing

If you're generating data faster than it can be uploaded:

```python
# Reduce batch sizes or flush more frequently
config = BufferConfig(
    flush_interval=2.0,  # Flush every 2 seconds instead of 5
    log_batch_size=50    # Smaller batches
)
```

### Uploads failing

Check the warnings in console output. Common issues:
- Network connectivity
- Authentication expired (run `ml-dash login`)
- Server unavailable

The buffering system will retry and warn you, but training continues.
