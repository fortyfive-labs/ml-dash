# Metrics

Track time-series metrics that change over time - loss, accuracy, learning rate, and custom measurements.

## Basic Usage

Log train and eval metrics together with a single call:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(prefix="alice/project/my-experiment").run as exp:

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model)
        eval_loss, eval_acc = evaluate(model)

        # Log all metrics for this epoch in one call
        exp.metrics.log(
            epoch=epoch,
            train=dict(loss=train_loss, accuracy=train_acc),
            eval=dict(loss=eval_loss, accuracy=eval_acc)
        )
```

## Alternative: Prefix-Based Logging

You can also log metrics using namespace prefixes with explicit context:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    for epoch in range(10):
        # Log train metrics with prefix
        exp.metrics("train").log(loss=train_loss, accuracy=train_acc)

        # Log eval metrics with prefix
        exp.metrics("eval").log(loss=eval_loss, accuracy=eval_acc)

        # Set epoch and flush
        exp.metrics.log(epoch=epoch).flush()
```

## Custom Metric Groups

Define your own metric groups beyond train/eval:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    # Log with custom groups
    exp.metrics.log(
        step=100,
        train=dict(loss=0.5, accuracy=0.85),
        eval=dict(loss=0.6, accuracy=0.82),
        system=dict(gpu_memory=4.2, cpu_percent=45),
        lr=dict(value=0.001)
    )
```

## Reading Data

Read metric data by index range:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:
    # Log data
    for i in range(100):
        exp.metrics("train").log(loss=1.0 / (i + 1), step=i)

    # Read first 10 points
    result = exp.metrics("train").read(start_index=0, limit=10)

    for point in result['data']:
        print(f"Index {point['index']}: {point['data']}")
```

## Buffer API

For high-frequency logging (e.g., per-batch), use the buffer API to accumulate values and periodically log summary statistics:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    for epoch in range(10):
        for batch in dataloader:
            loss = train_step(batch)
            acc = compute_accuracy(batch)

            # Buffer per-batch values (not written to disk yet)
            exp.metrics("train").buffer(loss=loss, accuracy=acc)

        # At end of epoch, compute and log summary statistics
        exp.metrics.buffer.log_summary()  # default: mean
        exp.metrics.log(epoch=epoch).flush()
```

This produces statistics like `loss.mean`, `accuracy.mean` per epoch.

### Buffer API Methods

- **`metrics("prefix").buffer(**kwargs)`** - Accumulate values for later summarization
- **`metrics.buffer.log_summary(*aggs)`** - Compute statistics and log to all prefixes
- **`metrics.buffer.peek(prefix, *keys, limit=5)`** - Non-destructive look at buffered values

### Supported Aggregations

```python
# Default (just mean)
exp.metrics.buffer.log_summary()

# Multiple aggregations
exp.metrics.buffer.log_summary("mean", "std", "min", "max", "count")

# Percentiles
exp.metrics.buffer.log_summary("p50", "p90", "p95", "p99")

# First/last values
exp.metrics.buffer.log_summary("first", "last")
```

Available aggregations: `mean`, `std`, `min`, `max`, `count`, `median`, `sum`, `p50`, `p90`, `p95`, `p99`, `first`, `last`

### Multiple Prefixes

Buffer works across multiple prefixes simultaneously:

```{code-block} python
:linenos:

for batch in dataloader:
    train_loss = train_step(batch)
    val_loss = validate_step(batch)

    # Buffer to different prefixes
    exp.metrics("train").buffer(loss=train_loss)
    exp.metrics("eval").buffer(loss=val_loss)

# Single call logs summaries for ALL prefixes
exp.metrics.buffer.log_summary("mean", "std")
```

## Summary Cache (Legacy API)

The summary cache API is still supported for backward compatibility:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:

    for epoch in range(10):
        for batch in dataloader:
            loss = train_step(batch)

            # Store per-batch values (not written to disk yet)
            exp.metrics("train").summary_cache.store(loss=loss)

        # At end of epoch, summarize and write to disk
        exp.metrics("train").summary_cache.set(epoch=epoch)
        exp.metrics("train").summary_cache.summarize()
```

### Summary Cache Methods

- **`store(**kwargs)`** - Accumulate values for later summarization
- **`set(**kwargs)`** - Set metadata (epoch, lr) that doesn't need aggregation
- **`summarize(clear=True)`** - Compute statistics and append to metrics
- **`peek(key, limit=None)`** - Non-destructive look at stored values

### Rolling vs Cumulative

```{code-block} python
:linenos:

# Rolling window (default) - clears cache after summarize
metric.summary_cache.summarize()  # clear=True by default

# Cumulative - keeps accumulating
metric.summary_cache.summarize(clear=False)
```

## Training Loop Example

```{code-block} python
:linenos:

with Experiment(prefix="alice/cv/mnist-training").run as exp:
    exp.params.set(learning_rate=0.001, batch_size=32)
    exp.log("Starting training")

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Log metrics
        exp.metrics("train").log(loss=train_loss)
        exp.metrics("eval").log(loss=val_loss, accuracy=val_accuracy)
        exp.metrics.log(epoch=epoch + 1).flush()

        exp.log(
            f"Epoch {epoch + 1}/10 complete",
            metadata={"train_loss": train_loss, "val_loss": val_loss}
        )

    exp.log("Training complete")
```

## Multiple Metrics in One Call

Combine related metrics:

```{code-block} python
:linenos:

with Experiment(prefix="alice/project/my-experiment").run as exp:
    for epoch in range(10):
        exp.metrics("train").log(
            epoch=epoch,
            loss=0.5 / (epoch + 1),
            accuracy=0.8 + epoch * 0.01
        )
        exp.metrics("eval").log(
            epoch=epoch,
            loss=0.6 / (epoch + 1),
            accuracy=0.75 + epoch * 0.01
        )
```

## Storage Format

**Local mode** - JSONL files:

```bash
cat .dash/alice/project/my-experiment/metrics/train/data.jsonl
```

```json
{"index": 0, "data": {"loss": 0.5, "epoch": 1}}
{"index": 1, "data": {"loss": 0.45, "epoch": 2}}
{"index": 2, "data": {"loss": 0.40, "epoch": 3}}
```

**Remote mode** - Two-tier storage:
- **Hot tier:** Recent data in MongoDB (fast access)
- **Cold tier:** Historical data in S3 (auto-archived after 10,000 points)

---

**Next:** Learn about [Files](files.md) to upload models, plots, and artifacts.
