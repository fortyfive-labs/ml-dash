"""
01 - Quickstart: Three ways to use ml-dash.

Shows the three usage styles:
  1. Context manager  (most common)
  2. Decorator
  3. Manual start / stop
"""

from ml_dash import Experiment, ml_dash_experiment

DASH_ROOT = "/tmp/ml-dash-demo"

# ---------------------------------------------------------------------------
# 1. Context manager  (recommended)
# ---------------------------------------------------------------------------
print("=== 1. Context manager ===")

with Experiment(
    prefix="alice/nlp/bert-finetune",
    dash_root=DASH_ROOT,
).run as exp:
    exp.log("Training started")
    exp.params.set(lr=3e-5, epochs=3, batch_size=16)
    for epoch in range(3):
        exp.metrics("train").log(loss=1.0 / (epoch + 1), epoch=epoch)
    exp.log("Training finished")
# Experiment is automatically marked COMPLETED on exit


# ---------------------------------------------------------------------------
# 2. Decorator style
# ---------------------------------------------------------------------------
print("=== 2. Decorator ===")


@ml_dash_experiment(
    prefix="alice/nlp/bert-finetune-decorator",
    dash_root=DASH_ROOT,
)
def train(experiment):
    # The decorator injects the active Experiment as 'experiment'
    experiment.log("Training via decorator")
    experiment.params.set(lr=1e-4, epochs=5)
    for epoch in range(5):
        experiment.metrics("train").log(loss=2.0 / (epoch + 1), epoch=epoch)


train()


# ---------------------------------------------------------------------------
# 3. Manual start / complete
# ---------------------------------------------------------------------------
print("=== 3. Manual start/complete ===")

exp = Experiment(prefix="alice/nlp/bert-manual", dash_root=DASH_ROOT)

exp.run.start()
try:
    exp.log("Doing work...")
    exp.params.set(lr=5e-5)
    exp.metrics("train").log(loss=0.42, step=0)
    exp.run.complete()
except Exception:
    exp.run.fail()   # marks the experiment FAILED


# ---------------------------------------------------------------------------
# 4. Environment-variable driven prefix
# ---------------------------------------------------------------------------
# You can set ML_DASH_PREFIX before running instead of passing prefix=:
#
#   export ML_DASH_PREFIX="alice/nlp/my-experiment"
#   python train.py
#
# Experiment() will pick it up automatically.

print("Done.")
