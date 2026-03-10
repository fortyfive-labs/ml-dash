"""
04 - Logs: Structured logging inside an experiment.

Covers:
  - exp.log()        – shorthand (level defaults to INFO)
  - exp.logs.info()
  - exp.logs.warn()
  - exp.logs.error()
  - exp.logs.debug()
  - exp.logs.fatal()
  - Extra metadata attached to a log entry
"""

from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

with Experiment(prefix="alice/nlp/logging-demo", dash_root=DASH_ROOT).run as exp:

    # ---------------------------------------------------------------------------
    # 1. Shorthand: exp.log() — INFO level by default
    # ---------------------------------------------------------------------------
    exp.log("Training started")
    exp.log("Epoch 1 complete", level="info")   # explicit level string

    # ---------------------------------------------------------------------------
    # 2. Structured log levels via exp.logs
    # ---------------------------------------------------------------------------
    exp.logs.info("Loading dataset from S3")
    exp.logs.debug("Batch shape: (64, 3, 224, 224)")
    exp.logs.warn("Learning rate scheduler not found, using constant LR")
    exp.logs.error("NaN detected in gradients at step 150")
    exp.logs.fatal("Out of memory — training aborted")

    # ---------------------------------------------------------------------------
    # 3. Extra metadata attached to a log entry
    #    Any keyword args become structured fields in the log record.
    # ---------------------------------------------------------------------------
    exp.logs.info(
        "Checkpoint saved",
        step=500,
        val_loss=0.312,
        path="checkpoints/epoch_5.pt",
    )

    exp.logs.error(
        "Validation spike detected",
        epoch=12,
        val_loss=1.87,
        prev_val_loss=0.33,
        action="reverting to best checkpoint",
    )
