"""
03 - Parameters: Storing and reading experiment hyperparameters.

Covers:
  - params.set()  – flat and nested dicts
  - params.log()  – alias for set()
  - params.get()  – read back (flat or nested)
  - Storing dataclass / params-proto objects
  - Updating params mid-run
"""

import dataclasses
from dataclasses import dataclass
from ml_dash import Experiment

DASH_ROOT = "/tmp/ml-dash-demo"

exp = Experiment(prefix="alice/vision/resnet-params", dash_root=DASH_ROOT)
exp.run.start()

# ---------------------------------------------------------------------------
# 1. Flat parameters
# ---------------------------------------------------------------------------
exp.params.set(
    lr=3e-4,
    batch_size=64,
    epochs=100,
    optimizer="adam",
    weight_decay=1e-5,
)

# ---------------------------------------------------------------------------
# 2. Nested parameters (stored as nested dict)
# ---------------------------------------------------------------------------
exp.params.set(
    model=dict(
        architecture="resnet50",
        pretrained=True,
        num_classes=1000,
    ),
    data=dict(
        dataset="imagenet",
        image_size=224,
        augmentation=dict(
            horizontal_flip=True,
            color_jitter=0.4,
        ),
    ),
)

# ---------------------------------------------------------------------------
# 3. params.log() is an alias for set()
# ---------------------------------------------------------------------------
exp.params.log(seed=42, run_id="abc123")

# ---------------------------------------------------------------------------
# 4. Storing a dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    nesterov: bool = True

cfg = TrainingConfig(lr=5e-4)
exp.params.set(training=dataclasses.asdict(cfg))   # convert to dict first

# ---------------------------------------------------------------------------
# 5. Read back params
# ---------------------------------------------------------------------------
# Flat (dot-separated keys)
flat = exp.params.get(flatten=True)
print("flat params:", flat)
# e.g. {"lr": 3e-4, "model.architecture": "resnet50", ...}

# Nested
nested = exp.params.get(flatten=False)
print("model architecture:", nested.get("model", {}).get("architecture"))

# ---------------------------------------------------------------------------
# 6. Update / overwrite params mid-run
#    Useful to record the actual best config after a sweep step.
# ---------------------------------------------------------------------------
exp.params.set(best_epoch=42, final_val_acc=0.768)

exp.run.complete()
