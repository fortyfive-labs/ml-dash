# ML-Dash `log_summary()` - Detailed Usage Guide

## Overview

`log_summary()` is a powerful feature for computing and logging statistics from buffered metric values. Instead of logging every single training step, you can **buffer** values and then compute aggregated statistics (mean, std, percentiles, etc.) in one call.

**Key Benefits:**
- ✅ Reduce metric logging overhead
- ✅ Automatic statistical aggregation
- ✅ Clean hierarchical naming (e.g., `loss.mean`, `loss.std`)
- ✅ Support for multiple prefixes (train/val/test)

---

## Basic Workflow

```python
from ml_dash import Experiment

exp = Experiment(prefix="user/project/exp1", dash_root=".dash")

with exp.run:
    # 1. Buffer values during training loop
    for step in range(100):
        loss = train_step()
        accuracy = evaluate()
        exp.metrics("train").buffer(loss=loss, accuracy=accuracy)

    # 2. Compute and log statistics
    exp.metrics.buffer.log_summary()  # Default: mean

    # 3. Log epoch number (non-buffered)
    exp.metrics.log(epoch=1)

    # 4. Flush to storage
    exp.metrics.flush()
```

**Result:** Instead of 100 data points, you get 1 summary with `loss.mean` and `accuracy.mean`.

---

## Supported Aggregations

### Basic Statistics

```python
exp.metrics.buffer.log_summary("mean")      # Average
exp.metrics.buffer.log_summary("std")       # Standard deviation
exp.metrics.buffer.log_summary("min")       # Minimum value
exp.metrics.buffer.log_summary("max")       # Maximum value
exp.metrics.buffer.log_summary("count")     # Number of values
exp.metrics.buffer.log_summary("sum")       # Sum of all values
exp.metrics.buffer.log_summary("median")    # Median (50th percentile)
```

### Percentiles

```python
exp.metrics.buffer.log_summary("p50")       # 50th percentile (median)
exp.metrics.buffer.log_summary("p90")       # 90th percentile
exp.metrics.buffer.log_summary("p95")       # 95th percentile
exp.metrics.buffer.log_summary("p99")       # 99th percentile
```

### First/Last Values

```python
exp.metrics.buffer.log_summary("first")     # First buffered value
exp.metrics.buffer.log_summary("last")      # Last buffered value
```

### Multiple Aggregations at Once

```python
# Compute multiple statistics in one call
exp.metrics.buffer.log_summary("mean", "std", "min", "max", "count")

# Common combinations
exp.metrics.buffer.log_summary("mean", "std")           # Mean ± std
exp.metrics.buffer.log_summary("mean", "p95")           # Mean + 95th percentile
exp.metrics.buffer.log_summary("min", "max", "median")  # Range summary
exp.metrics.buffer.log_summary("first", "last")         # Start and end values
```

**Output naming:** Each aggregation creates a new field with suffix:
- `loss` → `loss.mean`, `loss.std`, `loss.min`, `loss.max`

---

## Complete Examples

### Example 1: Training Loop with Per-Epoch Summaries

```python
from ml_dash import Experiment

exp = Experiment(prefix="user/mnist/training", dash_root=".dash")

with exp.run:
    for epoch in range(10):
        # Training phase - buffer metrics
        for batch in range(100):
            loss, acc = train_batch()
            exp.metrics("train").buffer(loss=loss, accuracy=acc)

        # Compute training statistics
        exp.metrics.buffer.log_summary("mean", "std", "min", "max")

        # Validation phase - buffer metrics
        for batch in range(20):
            val_loss, val_acc = validate_batch()
            exp.metrics("val").buffer(loss=val_loss, accuracy=val_acc)

        # Compute validation statistics
        exp.metrics.buffer.log_summary("mean", "std")

        # Log epoch number (not buffered)
        exp.metrics.log(epoch=epoch)

        # Flush all metrics to storage
        exp.metrics.flush()

# Result per epoch:
# - train/loss.mean, train/loss.std, train/loss.min, train/loss.max
# - train/accuracy.mean, train/accuracy.std, train/accuracy.min, train/accuracy.max
# - val/loss.mean, val/loss.std
# - val/accuracy.mean, val/accuracy.std
# - epoch (direct log)
```

### Example 2: Multiple Prefixes

```python
with exp.run:
    # Buffer to multiple prefixes
    for i in range(100):
        exp.metrics("train").buffer(loss=0.5 - i * 0.001, accuracy=0.8 + i * 0.001)
        exp.metrics("val").buffer(loss=0.6 - i * 0.001, accuracy=0.75 + i * 0.001)
        exp.metrics("test").buffer(accuracy=0.85 + i * 0.0005)

    # Single call computes summaries for ALL prefixes
    exp.metrics.buffer.log_summary("mean", "std", "p95")

    # Logs to:
    # - train: loss.mean, loss.std, loss.p95, accuracy.mean, accuracy.std, accuracy.p95
    # - val: loss.mean, loss.std, loss.p95, accuracy.mean, accuracy.std, accuracy.p95
    # - test: accuracy.mean, accuracy.std, accuracy.p95
```

### Example 3: Combining Buffered and Direct Logging

```python
with exp.run:
    for epoch in range(5):
        # Buffer step-level metrics
        for step in range(50):
            exp.metrics("train").buffer(loss=train_loss, grad_norm=grad_norm)

        # Log buffered statistics
        exp.metrics.buffer.log_summary("mean", "max")

        # Log non-statistical values directly (not buffered)
        exp.metrics.log(
            epoch=epoch,
            learning_rate=get_current_lr(),
            timestamp=time.time()
        )

        exp.metrics.flush()

# Epoch data will have:
# - loss.mean, loss.max (from buffer)
# - grad_norm.mean, grad_norm.max (from buffer)
# - epoch, learning_rate, timestamp (direct log)
```

### Example 4: Percentile Analysis

```python
with exp.run:
    # Collect 1000 inference latency measurements
    for i in range(1000):
        latency = measure_inference_latency()
        exp.metrics("performance").buffer(latency_ms=latency)

    # Log percentiles for latency analysis
    exp.metrics.buffer.log_summary("mean", "p50", "p90", "p95", "p99")

    exp.metrics.flush()

# Result:
# - latency_ms.mean  (average latency)
# - latency_ms.p50   (median - 50% of requests faster)
# - latency_ms.p90   (90% of requests faster)
# - latency_ms.p95   (95% of requests faster)
# - latency_ms.p99   (99% of requests faster)
```

### Example 5: Rolling Window Statistics

```python
with exp.run:
    for epoch in range(20):
        # Collect metrics for this epoch
        for step in range(100):
            exp.metrics("train").buffer(loss=loss, lr=lr)

        # Log summary for this epoch
        exp.metrics.buffer.log_summary("mean", "std")

        # Log epoch marker
        exp.metrics.log(epoch=epoch)
        exp.metrics.flush()

        # Buffer is automatically cleared after log_summary()
        # Next epoch starts fresh

# Each epoch gets its own summary statistics
```

---

## Advanced Features

### 1. Peek at Buffer (Non-Destructive)

```python
# Buffer some values
exp.metrics("train").buffer(loss=0.5, accuracy=0.8)
exp.metrics("train").buffer(loss=0.4, accuracy=0.85)

# Peek at buffered values (doesn't clear buffer)
buffered = exp.metrics.buffer.peek("train", "loss", limit=5)
print(buffered)  # {'loss': [0.5, 0.4]}

# Still available for log_summary()
exp.metrics.buffer.log_summary()
```

**Peek all prefixes:**
```python
buffered = exp.metrics.buffer.peek(limit=10)
# Returns: {'train/loss': [0.5, 0.4], 'train/accuracy': [0.8, 0.85], ...}
```

### 2. Automatic Buffer Clearing

```python
# First batch
for i in range(10):
    exp.metrics("train").buffer(loss=0.5)
exp.metrics.buffer.log_summary()  # Computes mean=0.5, clears buffer

# Second batch (starts fresh)
for i in range(10):
    exp.metrics("train").buffer(loss=0.3)
exp.metrics.buffer.log_summary()  # Computes mean=0.3 (not mixed with first batch)
```

### 3. Handling None/NaN Values

```python
# None values are converted to NaN and filtered out
exp.metrics("train").buffer(loss=0.5, accuracy=None)
exp.metrics("train").buffer(loss=0.4, accuracy=0.85)

exp.metrics.buffer.log_summary("mean", "count")

# Result:
# - loss.mean = 0.45 (mean of [0.5, 0.4])
# - loss.count = 2
# - accuracy.mean = 0.85 (None filtered out)
# - accuracy.count = 1
```

### 4. Empty Buffer Handling

```python
# If buffer is empty, log_summary() does nothing (no error)
exp.metrics.buffer.log_summary()  # No-op if no values buffered
```

---

## Common Patterns

### Pattern 1: Per-Epoch Training Statistics

```python
with exp.run:
    for epoch in range(num_epochs):
        # Training
        for batch_idx, (data, target) in enumerate(train_loader):
            loss, acc = train_step(data, target)
            exp.metrics("train").buffer(loss=loss, accuracy=acc)

        # Log training summary
        exp.metrics.buffer.log_summary("mean", "std")

        # Validation
        for data, target in val_loader:
            val_loss, val_acc = validate(data, target)
            exp.metrics("val").buffer(loss=val_loss, accuracy=val_acc)

        # Log validation summary
        exp.metrics.buffer.log_summary("mean")

        # Log epoch-level metadata
        exp.metrics.log(epoch=epoch, lr=scheduler.get_last_lr()[0])
        exp.metrics.flush()
```

### Pattern 2: Gradient Monitoring

```python
with exp.run:
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = train_step(batch)

            # Monitor gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    exp.metrics("gradients").buffer(**{f"{name}_norm": grad_norm})

        # Log gradient statistics
        exp.metrics.buffer.log_summary("mean", "max", "p95")
        exp.metrics.log(epoch=epoch)
        exp.metrics.flush()
```

### Pattern 3: A/B Testing Results

```python
with exp.run:
    # Run experiment A
    for trial in range(100):
        result = run_experiment_a()
        exp.metrics("experiment_a").buffer(success_rate=result)

    # Run experiment B
    for trial in range(100):
        result = run_experiment_b()
        exp.metrics("experiment_b").buffer(success_rate=result)

    # Compare statistics
    exp.metrics.buffer.log_summary("mean", "std", "p50", "p95")
    exp.metrics.flush()

# Easily compare:
# - experiment_a/success_rate.mean vs experiment_b/success_rate.mean
# - experiment_a/success_rate.std vs experiment_b/success_rate.std
```

---

## API Reference

### `exp.metrics.buffer.log_summary(*aggs)`

**Parameters:**
- `*aggs` (str): Aggregation function names. If empty, defaults to `("mean",)`.

**Supported Aggregations:**
- `"mean"` - Average of values
- `"std"` - Standard deviation (requires ≥2 values, otherwise 0.0)
- `"min"` - Minimum value
- `"max"` - Maximum value
- `"count"` - Number of values
- `"median"` - Median value (same as p50)
- `"sum"` - Sum of all values
- `"p50"` - 50th percentile
- `"p90"` - 90th percentile
- `"p95"` - 95th percentile
- `"p99"` - 99th percentile
- `"first"` - First value in buffer
- `"last"` - Last value in buffer

**Returns:** None

**Side Effects:**
- Computes statistics for all buffered values across all prefixes
- Logs computed statistics using hierarchical naming (e.g., `loss.mean`)
- Clears all buffers after logging

**Raises:**
- `ValueError`: If unsupported aggregation function is specified

---

## Best Practices

### ✅ DO:

1. **Use buffering for high-frequency metrics**
   ```python
   for step in range(1000):
       exp.metrics("train").buffer(loss=loss)  # Good
   exp.metrics.buffer.log_summary()
   ```

2. **Combine multiple aggregations**
   ```python
   exp.metrics.buffer.log_summary("mean", "std", "min", "max")
   ```

3. **Use different prefixes for different phases**
   ```python
   exp.metrics("train").buffer(...)
   exp.metrics("val").buffer(...)
   exp.metrics.buffer.log_summary()  # Logs both
   ```

4. **Clear separation of buffered and direct logs**
   ```python
   exp.metrics("train").buffer(loss=0.5)  # Buffered
   exp.metrics.buffer.log_summary()
   exp.metrics.log(epoch=1)  # Direct (not buffered)
   ```

### ❌ DON'T:

1. **Don't buffer infrequent values**
   ```python
   # Bad - epoch is logged once, no need to buffer
   exp.metrics.buffer(epoch=1)
   exp.metrics.buffer.log_summary()

   # Good - log directly
   exp.metrics.log(epoch=1)
   ```

2. **Don't forget to flush**
   ```python
   exp.metrics.buffer.log_summary()
   # Missing: exp.metrics.flush()  # Data might not be saved!
   ```

3. **Don't mix statistical and non-statistical values in buffer**
   ```python
   # Bad - timestamp doesn't need statistics
   exp.metrics.buffer(loss=0.5, timestamp=time.time())

   # Good - separate them
   exp.metrics.buffer(loss=0.5)
   exp.metrics.buffer.log_summary()
   exp.metrics.log(timestamp=time.time())
   ```

---

## Error Handling

### Invalid Aggregation

```python
try:
    exp.metrics.buffer.log_summary("invalid_agg")
except ValueError as e:
    print(e)  # "Unsupported aggregation: invalid_agg. Supported: {...}"
```

### Empty Buffer (No Error)

```python
# This is safe - does nothing if buffer is empty
exp.metrics.buffer.log_summary()
```

### Non-Numeric Values (Silently Skipped)

```python
exp.metrics("train").buffer(loss="not_a_number")  # Silently skipped
exp.metrics("train").buffer(loss=0.5)  # This one works
exp.metrics.buffer.log_summary()  # Only computes from 0.5
```

---

## Performance Considerations

### Memory Usage

Buffered values are stored in memory until `log_summary()` is called:

```python
# 10,000 buffered values = ~80KB memory (approximate)
for i in range(10000):
    exp.metrics("train").buffer(loss=i, accuracy=i/100)

# Clear buffer and log statistics
exp.metrics.buffer.log_summary()  # Frees memory
```

**Recommendation:** Call `log_summary()` periodically (e.g., once per epoch) to avoid excessive memory usage.

### Network Efficiency

```python
# Bad - 1000 network calls
for i in range(1000):
    exp.metrics.log(loss=i)

# Good - 1 network call with aggregated statistics
for i in range(1000):
    exp.metrics.buffer(loss=i)
exp.metrics.buffer.log_summary("mean", "std", "min", "max")
```

---

## Migration from Legacy SummaryCache

If you're using the old `SummaryCache` API, here's how to migrate:

```python
# Old API (still supported)
from ml_dash import Experiment
exp = Experiment(...)
with exp.run:
    exp.summary_cache.collect("train/loss", 0.5)
    exp.summary_cache.summarize("train/loss")

# New API (recommended)
from ml_dash import Experiment
exp = Experiment(...)
with exp.run:
    exp.metrics("train").buffer(loss=0.5)
    exp.metrics.buffer.log_summary()
```

**Advantages of new API:**
- Cleaner syntax with metric prefixes
- Automatic multi-prefix handling
- More aggregation options
- Better integration with MetricsManager

---

## Summary

`log_summary()` is essential for efficient metric logging in ML-Dash:

1. **Buffer** values during training: `exp.metrics("train").buffer(loss=0.5)`
2. **Compute statistics**: `exp.metrics.buffer.log_summary("mean", "std")`
3. **Log direct values**: `exp.metrics.log(epoch=1)`
4. **Flush to storage**: `exp.metrics.flush()`

This pattern reduces logging overhead while providing rich statistical insights about your training process.
