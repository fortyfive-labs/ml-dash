# Metrics

Track time-series metrics that change over time - loss, accuracy, learning rate, and custom measurements.

## Basic Usage

Log train and eval metrics together with a single call:

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model)
        eval_loss, eval_acc = evaluate(model)

        # Log all metrics for this epoch in one call
        experiment.metrics.log(
            epoch=epoch,
            train=dict(loss=train_loss, accuracy=train_acc),
            eval=dict(loss=eval_loss, accuracy=eval_acc)
        )
```

## Alternative: Prefix-Based Logging

You can also log metrics using namespace prefixes with explicit context:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:

    for epoch in range(10):
        # Log train metrics with prefix
        experiment.metrics("train").log(loss=train_loss, accuracy=train_acc)

        # Log eval metrics with prefix
        experiment.metrics("eval").log(loss=eval_loss, accuracy=eval_acc)

        # Set epoch and flush
        experiment.metrics.log(epoch=epoch).flush()
```

## Custom Metric Groups

Define your own metric groups beyond train/eval:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:

    # Log with custom groups
    experiment.metrics.log(
        step=100,
        train=dict(loss=0.5, accuracy=0.85),
        eval=dict(loss=0.6, accuracy=0.82),
        system=dict(gpu_memory=4.2, cpu_percent=45),
        lr=dict(value=0.001)
    )
```

## Batch Append

Append multiple data points at once for better performance:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:
    result = experiment.metrics("train_loss").append_batch([
        {"value": 0.5, "step": 1, "epoch": 1},
        {"value": 0.45, "step": 2, "epoch": 1},
        {"value": 0.40, "step": 3, "epoch": 1},
        {"value": 0.38, "step": 4, "epoch": 1},
    ])

    print(f"Appended {result['count']} points")
```

## Reading Data

Read metric data by index range:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:
    # Append data
    for i in range(100):
        experiment.metrics("loss").append(value=1.0 / (i + 1), step=i)

    # Read first 10 points
    result = experiment.metrics("loss").read(start_index=0, limit=10)

    for point in result['data']:
        print(f"Index {point['index']}: {point['data']}")
```

## Summary Cache

For high-frequency logging (e.g., per-batch), use the summary cache to store values and periodically summarize them into statistics:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:

    for epoch in range(10):
        for batch in dataloader:
            loss = train_step(batch)

            # Store per-batch values (not written to disk yet)
            experiment.metrics("train").summary_cache.store(loss=loss)

        # At end of epoch, summarize and write to disk
        experiment.metrics("train").summary_cache.set(epoch=epoch)
        experiment.metrics("train").summary_cache.summarize()
```

This produces statistics like `loss.mean`, `loss.min`, `loss.max`, `loss.std`, `loss.count` per epoch.

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

with Experiment(name="mnist-training", project="cv",
        local_path=".dash").run as experiment:
    experiment.params.set(learning_rate=0.001, batch_size=32)
    experiment.log("Starting training")

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Metric metrics
        experiment.metrics.log(epoch=epoch + 1)
        experiment.metrics("train").log(loss=train_loss, accuracy=val_accuracy)
        experiment.metrics("eval").log(loss=val_loss, accuracy=val_accuracy)
        experiment.metrics.flush()

        experiment.log(
            f"Epoch {epoch + 1}/10 complete",
            metadata={"train_loss": train_loss, "val_loss": val_loss}
        )

    experiment.log("Training complete")
```

## Batch Collection Pattern

Collect points in memory, then append in batches:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:
    batch = []

    for step in range(1000):
        loss = train_step()

        batch.append({"value": loss, "step": step, "epoch": step // 100})

        # Append every 100 steps
        if len(batch) >= 100:
            experiment.metrics("train_loss").append_batch(batch)
            batch = []

    # Append remaining
    if batch:
        experiment.metrics("train_loss").append_batch(batch)
```

## Multiple Metrics in One Metric

Combine related metrics:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".dash").run as experiment:
    for epoch in range(10):
        experiment.metrics("all_metrics").append(
            epoch=epoch,
            train_loss=0.5 / (epoch + 1),
            val_loss=0.6 / (epoch + 1),
            train_acc=0.8 + epoch * 0.01,
            val_acc=0.75 + epoch * 0.01
        )
```

## Storage Format

**Local mode** - JSONL files:

```bash
cat ./experiments/project/my-experiment/metrics/train_loss/data.jsonl
```

```json
{"index": 0, "data": {"value": 0.5, "epoch": 1}}
{"index": 1, "data": {"value": 0.45, "epoch": 2}}
{"index": 2, "data": {"value": 0.40, "epoch": 3}}
```

**Remote mode** - Two-tier storage:
- **Hot tier:** Recent data in MongoDB (fast access)
- **Cold tier:** Historical data in S3 (auto-archived after 10,000 points)

---

**Next:** Learn about [Files](files.md) to upload models, plots, and artifacts.
