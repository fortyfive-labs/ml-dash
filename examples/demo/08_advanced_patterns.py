"""
08 - Advanced Patterns: Real training loop integrations.

Covers:
  - Complete training loop using the documented pattern
  - Background buffering (flush control)
  - SummaryCache rolling-window in a large training loop
  - MetricsManager.buffer for batch aggregation with multiple stats
  - Experiment run states (complete / fail / cancel)
  - exp.flush()  – manual flush of all buffered data
  - Write-protect an experiment after creation
  - Duplicate (copy) a file within an experiment
"""

import math
import random
from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

# ---------------------------------------------------------------------------
# 1. Canonical training loop pattern
#    Recommended way to log per-epoch metrics with a global epoch counter.
# ---------------------------------------------------------------------------
print("=== 1. Canonical training loop ===")

with Experiment(
    prefix="alice/vision/canonical-loop",
    dash_root=DASH_ROOT,
).run as exp:
    exp.params.set(lr=1e-3, epochs=5, batch_size=32)

    for epoch in range(5):
        # Simulate per-batch metrics
        train_loss = 1.0 - epoch * 0.15 + random.uniform(-0.05, 0.05)
        train_acc  = 0.5 + epoch * 0.08 + random.uniform(-0.02, 0.02)
        eval_loss  = 1.1 - epoch * 0.13 + random.uniform(-0.05, 0.05)
        eval_acc   = 0.45 + epoch * 0.07 + random.uniform(-0.02, 0.02)

        # Log named metrics per epoch
        exp.metrics("train").log(loss=train_loss, accuracy=train_acc)
        exp.metrics("eval").log(loss=eval_loss, accuracy=eval_acc)

        # Log the shared epoch counter and flush
        exp.metrics.log(epoch=epoch).flush()

    exp.params.set(final_val_acc=eval_acc)   # record best metric in params


# ---------------------------------------------------------------------------
# 2. SummaryCache rolling window over batches
# ---------------------------------------------------------------------------
print("=== 2. SummaryCache rolling window ===")

with Experiment(
    prefix="alice/vision/summary-cache-loop",
    dash_root=DASH_ROOT,
).run as exp:
    train_m = exp.metrics("train")
    eval_m  = exp.metrics("eval")
    log_every = 20   # log a summary point every 20 batches

    for epoch in range(3):
        for batch_idx in range(100):
            step = epoch * 100 + batch_idx
            train_loss = math.exp(-step * 0.01) + random.gauss(0, 0.02)
            train_acc  = 1 - math.exp(-step * 0.008) + random.gauss(0, 0.01)

            # Accumulate raw values
            train_m.summary_cache.store(loss=train_loss, accuracy=train_acc)

            if (batch_idx + 1) % log_every == 0:
                # Attach contextual metadata, then emit a summary data point
                train_m.summary_cache.set(
                    epoch=epoch,
                    step=step,
                    lr=1e-3 * (0.95 ** epoch),
                )
                train_m.summary_cache.summarize()
                # Logged fields: loss.mean, loss.std, loss.min, loss.max,
                #                accuracy.mean, ..., epoch, step, lr

        # Eval summary once per epoch
        eval_loss = math.exp(-epoch * 0.5) * 1.1
        eval_m.summary_cache.store(loss=eval_loss)
        eval_m.summary_cache.set(epoch=epoch)
        eval_m.summary_cache.summarize()

    exp.flush()


# ---------------------------------------------------------------------------
# 3. MetricsManager.buffer with multiple aggregation stats
# ---------------------------------------------------------------------------
print("=== 3. MetricsManager.buffer with stats ===")

with Experiment(
    prefix="alice/vision/buffer-stats",
    dash_root=DASH_ROOT,
).run as exp:
    for epoch in range(3):
        for _ in range(50):
            loss = random.gauss(1.0 - epoch * 0.2, 0.1)
            exp.metrics("train").buffer(loss=loss)
            exp.metrics("eval").buffer(loss=loss + 0.1)

        # Log mean + std + p95 for both metrics simultaneously
        exp.metrics.buffer.log_summary("mean", "std", "p95")
        # Result: {"train.loss.mean": ..., "train.loss.std": ...,
        #          "eval.loss.mean": ..., ...}

    exp.flush()


# ---------------------------------------------------------------------------
# 4. Handling run states
# ---------------------------------------------------------------------------
print("=== 4. Run states ===")

# COMPLETED (normal)
exp_ok = Experiment(prefix="alice/states/completed", dash_root=DASH_ROOT)
exp_ok.run.start()
exp_ok.metrics("train").log(loss=0.3)
exp_ok.run.complete()

# FAILED (exception during training)
exp_fail = Experiment(prefix="alice/states/failed", dash_root=DASH_ROOT)
exp_fail.run.start()
try:
    raise ValueError("Simulated training failure")
except ValueError:
    exp_fail.logs.error("Training failed with ValueError")
    exp_fail.run.fail()

# CANCELLED (user interruption)
exp_cancel = Experiment(prefix="alice/states/cancelled", dash_root=DASH_ROOT)
exp_cancel.run.start()
exp_cancel.run.cancel()

# Using context manager — automatically calls fail() on exception
try:
    with Experiment(prefix="alice/states/auto-fail", dash_root=DASH_ROOT).run as exp:
        exp.metrics("train").log(loss=0.5)
        raise RuntimeError("Something went wrong")
except RuntimeError:
    pass   # exp.run.fail() was called automatically


# ---------------------------------------------------------------------------
# 5. exp.flush()  – explicit flush of all buffered data
#    Important in remote mode to ensure data is sent before reading back.
# ---------------------------------------------------------------------------
print("=== 5. Manual flush ===")

with Experiment(
    prefix="alice/flush-demo",
    dash_root=DASH_ROOT,
).run as exp:
    for i in range(10):
        exp.metrics("train").log(loss=1.0 / (i + 1))

    exp.flush()   # ensures all queued writes are persisted


print("Done.")
