# Metrics

Metric time-series metrics that change over time - loss, accuracy, learning rate, and custom measurements. Metrics support flexible schemas that you define.

## Basic Usage

```{code-block} python
:linenos:

from ml_dash import Experiment

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Append a single data point
    experiment.metrics("train_loss").append(value=0.5, epoch=1)

    # With step and epoch
    experiment.metrics("accuracy").append(value=0.85, step=100, epoch=1)
```

## Flexible Schema

Define your own data structure for each metric:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
    # Simple value
    experiment.metrics("loss").append(value=0.5)

    # Multiple fields per point
    experiment.metrics("metrics").append(
        loss=0.5,
        accuracy=0.85,
        learning_rate=0.001,
        epoch=1
    )

    # With timestamp
    import time
    experiment.metrics("system").append(
        cpu_percent=45.2,
        memory_mb=1024,
        timestamp=time.time()
    )
```

## Batch Append

Append multiple data points at once for better performance:

```{code-block} python
:linenos:

with Experiment(name="my-experiment", project="project",
        local_path=".ml-dash").run as experiment:
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
        local_path=".ml-dash").run as experiment:
    # Append data
    for i in range(100):
        experiment.metrics("loss").append(value=1.0 / (i + 1), step=i)

    # Read first 10 points
    result = experiment.metrics("loss").read(start_index=0, limit=10)

    for point in result['data']:
        print(f"Index {point['index']}: {point['data']}")
```

## Training Loop Example

```{code-block} python
:linenos:

with Experiment(name="mnist-training", project="cv",
        local_path=".ml-dash").run as experiment:
    experiment.params.set(learning_rate=0.001, batch_size=32)
    experiment.log("Starting training")

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_accuracy = validate(model, val_loader)

        # Metric metrics
        experiment.metrics("train_loss").append(value=train_loss, epoch=epoch + 1)
        experiment.metrics("val_loss").append(value=val_loss, epoch=epoch + 1)
        experiment.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch + 1)

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
        local_path=".ml-dash").run as experiment:
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
        local_path=".ml-dash").run as experiment:
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
