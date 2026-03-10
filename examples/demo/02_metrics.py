"""
02 - Metrics: All ways to log, aggregate, and read metrics.

Covers:
  - Named metrics with .log()
  - Fluent chaining pattern
  - Nested-dict pattern (log to multiple metrics at once)
  - MetricsManager.buffer + log_summary()  (batch aggregation)
  - SummaryCache  (rolling window statistics)
  - .read() and .stats()
  - .list_all()
"""

from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

exp = Experiment(prefix="alice/vision/resnet-metrics", dash_root=DASH_ROOT)
exp.run.start()

# ---------------------------------------------------------------------------
# 1. Basic named metric log
# ---------------------------------------------------------------------------
exp.metrics("train").log(loss=0.91, accuracy=0.62, step=0)
exp.metrics("train").log(loss=0.74, accuracy=0.71, step=1)
exp.metrics("eval").log(loss=0.80, accuracy=0.68, step=1)

# ---------------------------------------------------------------------------
# 2. Fluent chaining
#    experiment.metrics("name").log(...) returns self -> chainable
# ---------------------------------------------------------------------------
exp.metrics("train").log(loss=0.60).log(loss=0.55)   # two points in one line

# ---------------------------------------------------------------------------
# 3. Unnamed root metric   (goes to metric name "None")
# ---------------------------------------------------------------------------
exp.metrics.log(epoch=1)          # simple key-value
exp.metrics.log(epoch=2).flush()  # chain a flush at the end of an epoch

# ---------------------------------------------------------------------------
# 4. Nested-dict pattern
#    Scalar keys -> root metric;  dict values -> sub-metrics
# ---------------------------------------------------------------------------
exp.metrics.log(
    epoch=3,
    train=dict(loss=0.48, accuracy=0.80),
    eval=dict(loss=0.52, accuracy=0.76),
)
# This logs:
#   metric "None"  <- {epoch: 3}
#   metric "train" <- {epoch: 3, loss: 0.48, accuracy: 0.80}
#   metric "eval"  <- {epoch: 3, loss: 0.52, accuracy: 0.76}

# ---------------------------------------------------------------------------
# 5. metrics.buffer + log_summary  (accumulate many values, log statistics)
# ---------------------------------------------------------------------------
# Accumulate batch losses during a training loop
for batch_loss in [0.5, 0.48, 0.51, 0.47, 0.49]:
    exp.metrics("train").buffer(loss=batch_loss, accuracy=0.80)
    exp.metrics("eval").buffer(loss=batch_loss + 0.05)

# Compute and log mean (default), optionally std / p95 / min / max
exp.metrics.buffer.log_summary()                         # logs .mean for each prefix
exp.metrics.buffer.log_summary("mean", "std", "min", "max")  # explicit aggregations

# ---------------------------------------------------------------------------
# 6. SummaryCache  (rolling window inside a training loop)
# ---------------------------------------------------------------------------
train_metric = exp.metrics("train_sc")
eval_metric = exp.metrics("eval_sc")

log_every = 10
for batch_idx in range(50):
    train_loss = 1.0 - batch_idx * 0.015
    eval_loss  = 1.1 - batch_idx * 0.012

    train_metric.summary_cache.store(loss=train_loss, accuracy=0.5 + batch_idx * 0.01)
    eval_metric.summary_cache.store(loss=eval_loss)

    if (batch_idx + 1) % log_every == 0:
        train_metric.summary_cache.set(lr=0.001, epoch=batch_idx // log_every)
        train_metric.summary_cache.summarize()   # logs loss.mean, loss.std, etc.
        eval_metric.summary_cache.summarize()

exp.flush()   # flush buffered data to remote / local storage

# ---------------------------------------------------------------------------
# 7. Adding description, tags, and metadata to a metric
# ---------------------------------------------------------------------------
exp.metrics(
    "val_loss",
    description="Validation cross-entropy loss",
    tags=["validation", "loss"],
    metadata={"dataset": "imagenet", "split": "val"},
).log(loss=0.34)

# ---------------------------------------------------------------------------
# 8. Reading back metrics
# ---------------------------------------------------------------------------
# Read data points
result = exp.metrics("train").read(start_index=0, limit=100)
print(f"train metric has {result['total']} data points")
for point in result["data"]:
    print(f"  index={point['index']}  data={point['data']}")

# Stats (counts, timestamps)
stats = exp.metrics("train").stats()
print(f"totalDataPoints: {stats['totalDataPoints']}")

# List all metrics in this experiment
all_metrics = exp.metrics("train").list_all()
print("All metrics:", [m["name"] for m in all_metrics])

exp.run.complete()
