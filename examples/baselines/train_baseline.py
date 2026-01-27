"""
Baseline Training Script with Static Paths

This script is for systematic baseline experiments that should have
STATIC paths (no datetime) for permanent reference and comparison.

Usage:
    # Single baseline run
    python train_baseline.py --train.learning-rate 0.01

    # Run from launcher
    python launch_baseline.py --sweep resnet_baseline.jsonl

Path structure (NO datetime):
    {namespace}/ml-experiments/baselines/resnet18/001
    {namespace}/ml-experiments/baselines/resnet18/002
"""
import os
import random
import time
from datetime import datetime

from params_proto import proto
from ml_dash.run import RUN
from ml_dash import userinfo


@proto.prefix
class Train:
    """Training hyperparameters."""
    learning_rate: float = 0.01  # Learning rate
    batch_size: int = 32  # Batch size
    optimizer: str = "SGD"  # Optimizer: SGD or Adam
    momentum: float = 0.9  # Momentum for SGD
    epochs: int = 10  # Number of epochs
    weight_decay: float = 5e-4  # Weight decay


@proto.prefix
class Model:
    """Model architecture settings."""
    name: str = "ResNet-18"  # Model architecture
    dropout: float = 0.0  # Dropout rate
    pretrained: bool = False  # Use pretrained weights


@proto.prefix
class Eval:
    """Evaluation configuration."""
    metric: str = "accuracy"  # Primary metric
    dataset: str = "CIFAR-10"  # Dataset name
    test_batch_size: int = 100  # Test batch size


# Sweep metadata
sweep_index: int = 0  # Index in sweep
sweep_id: str = "resnet_baseline"  # Sweep identifier


@proto.cli
def main():
    """Run baseline training experiment with static path."""

    # Configure ML-Dash from RUN singleton
    if not hasattr(RUN, 'owner') or RUN.owner == os.environ.get('USER'):
        owner = userinfo.username
        if owner:
            RUN.owner = owner

    # Set entry point for path detection
    RUN.entry = __file__

    # CRITICAL: For baselines, set static prefix (no datetime)
    # This ensures baselines always appear at the same path
    baseline_name = Model.name.lower().replace("-", "")  # "resnet-18" -> "resnet18"
    RUN.prefix = f"{RUN.owner}/ml-experiments/baselines/{baseline_name}"

    # Handle sweep coordination if launched from sweep launcher
    sweep_job_counter = os.environ.get("ML_DASH_JOB_COUNTER")
    if sweep_job_counter:
        RUN.job_counter = int(sweep_job_counter)

    # Print configuration
    config_summary = f"""
{'='*80}
BASELINE TRAINING CONFIGURATION
{'='*80}

📊 Training:
   Learning Rate: {Train.learning_rate}
   Batch Size:    {Train.batch_size}
   Optimizer:     {Train.optimizer}
   Momentum:      {Train.momentum}
   Epochs:        {Train.epochs}
   Weight Decay:  {Train.weight_decay}

🏗️  Model:
   Architecture:  {Model.name}
   Dropout:       {Model.dropout}
   Pretrained:    {Model.pretrained}

📈 Evaluation:
   Metric:        {Eval.metric}
   Dataset:       {Eval.dataset}
   Test Batch:    {Eval.test_batch_size}

🔧 ML-Dash:
   Owner:         {RUN.owner}
   Project:       {RUN.project}
   Prefix:        {RUN.prefix} (STATIC - no datetime)
   Baseline:      {sweep_index + 1} (ID: {sweep_id})
"""

    if sweep_job_counter:
        config_summary += f"   Job Counter:   {RUN.job_counter:03d}\n"

    config_summary += f"\n{'='*80}\n"
    print(config_summary)

    # Import dxp after RUN configuration
    from ml_dash.auto_start import dxp

    print(f"Baseline Path: {dxp.run.prefix}/{RUN.job_counter:03d}\n")
    print(f"{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    # Run experiment
    with dxp.run:
        dxp.log(f"🔷 Baseline Run {sweep_index + 1} (ID: {sweep_id})", level="info")

        # Log all config groups
        dxp.params.set(**{
            **{f"train/{k}": v for k, v in Train._dict.items()},
            **{f"model/{k}": v for k, v in Model._dict.items()},
            **{f"eval/{k}": v for k, v in Eval._dict.items()},
            "sweep_index": sweep_index,
            "sweep_id": sweep_id,
            "is_baseline": True,
        })

        # Simulate training
        lr_factor = {0.1: 0.85, 0.01: 1.0, 0.001: 0.95, 0.0001: 0.80}.get(Train.learning_rate, 0.9)
        batch_factor = {32: 1.0, 64: 0.98, 128: 0.95}.get(Train.batch_size, 0.95)
        opt_factor = 1.0 if Train.optimizer == "Adam" else 0.97
        perf_factor = lr_factor * batch_factor * opt_factor

        best_val_acc = 0.0

        dxp.log(f"Training baseline for {Train.epochs} epochs", level="info")
        for epoch in range(1, Train.epochs + 1):
            train_loss = (2.5 * (0.65 ** epoch) / perf_factor) + random.uniform(0, 0.1)
            train_acc = min(0.98, (0.3 + epoch * 0.13) * perf_factor + random.uniform(0, 0.03))
            val_loss = (2.3 * (0.65 ** epoch) / perf_factor) + random.uniform(0, 0.15)
            val_acc = min(0.95, (0.25 + epoch * 0.13) * perf_factor + random.uniform(0, 0.03))
            best_val_acc = max(best_val_acc, val_acc)

            dxp.metrics("train").log(epoch=epoch, loss=train_loss, accuracy=train_acc)
            dxp.metrics("val").log(epoch=epoch, loss=val_loss, accuracy=val_acc)

            print(f"  Epoch {epoch:2d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_val_acc:.4f}")
            time.sleep(0.2)

        dxp.metrics("summary").log(
            best_val_accuracy=best_val_acc,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
        )

        dxp.log(f"✅ Baseline training complete! Best {Eval.metric}: {best_val_acc:.4f}", level="info")

    result_summary = f"""
{'='*80}
✓ BASELINE TRAINING COMPLETE
{'='*80}

Best Validation Accuracy: {best_val_acc:.4f}

View results: https://dash.ml/{RUN.owner}/{RUN.project}
Baseline path: {RUN.prefix}/{RUN.job_counter:03d}
{'='*80}
"""
    print(result_summary)


if __name__ == "__main__":
    main()
