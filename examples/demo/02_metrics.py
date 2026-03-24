"""
02 - Metrics: All ways to log, aggregate, and read metrics.

Covers:
  - Named metrics with .log()
  - Fluent chaining pattern
  - Nested-dict pattern (log to multiple metrics at once)
  - MetricBuilder.buffer()  (per-metric buffering)
  - MetricsManager.buffer + log_summary()  (cross-metric batch aggregation)
  - All log_summary() aggregations: mean, std, min, max, count, median, sum,
                                     p50, p90, p95, p99, last, first
  - MetricsManager.buffer.peek()  (inspect buffered values without flushing)
  - SummaryCache  (rolling window statistics)
  - SummaryCache.peek()   (inspect cache without flushing)
  - SummaryCache.summarize(clear=False)  (cumulative/non-clearing mode)
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
# 5a. MetricBuilder.buffer()  — per-metric buffering via the MetricBuilder
#     Values are accumulated inside the MetricsManager's BufferManager.
# ---------------------------------------------------------------------------
for batch_loss in [0.5, 0.48, 0.51, 0.47, 0.49]:
    exp.metrics("train").buffer(loss=batch_loss, accuracy=0.80)
    exp.metrics("eval").buffer(loss=batch_loss + 0.05)

# ---------------------------------------------------------------------------
# 5b. metrics.buffer.peek()  — inspect buffered values without consuming them
# ---------------------------------------------------------------------------
buffered = exp.metrics.buffer.peek()
print("Buffered so far:", buffered)
# e.g. {'train/loss': [0.5, 0.48, ...], 'eval/loss': [...]}

buffered_train = exp.metrics.buffer.peek("train")
print("train buffer:", buffered_train)

# ---------------------------------------------------------------------------
# 5c. MetricsManager.buffer.log_summary()  — compute stats and log
# ---------------------------------------------------------------------------
# Default: mean only
exp.metrics.buffer.log_summary()

# Re-fill buffer for additional aggregations demo
for batch_loss in [0.5, 0.48, 0.51, 0.47, 0.49]:
    exp.metrics("train").buffer(loss=batch_loss)

# All supported aggregations:
#   mean, std, min, max, count, median, sum, p50, p90, p95, p99, last, first
exp.metrics.buffer.log_summary("mean", "std", "min", "max", "count", "p95", "last")

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
        # peek() — inspect what's buffered without consuming it
        current = train_metric.summary_cache.peek("loss", limit=3)
        print(f"  batch {batch_idx}: last 3 losses = {current.get('loss', [])}")

        train_metric.summary_cache.set(lr=0.001, epoch=batch_idx // log_every)
        train_metric.summary_cache.summarize()         # rolling: clears after logging
        eval_metric.summary_cache.summarize()

# summarize(clear=False) — cumulative mode: keeps values after logging
# Useful for running overall statistics without resetting the window
cumulative_metric = exp.metrics("cumulative")
for v in [0.9, 0.8, 0.7, 0.6]:
    cumulative_metric.summary_cache.store(loss=v)
cumulative_metric.summary_cache.summarize(clear=False)   # values still in cache
cumulative_metric.summary_cache.store(loss=0.5)
cumulative_metric.summary_cache.summarize(clear=False)   # now includes all 5 values

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
