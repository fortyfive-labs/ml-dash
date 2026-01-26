"""
ResNet Training Script for Hyperparameter Sweep (using params-proto)

This script uses params-proto for hyperparameter management, providing:
- Clean class-based parameter definition with @proto.prefix decorator
- Type hints and defaults
- Automatic CLI parsing (use --config.param-name value)
- Easy integration with ML-Dash (one-line logging)

Usage:
    # Example 1: Run with auto-detected namespace (recommended)
    python3 train.py
    # Uses your authenticated user's namespace automatically

    # Example 2: Run experiments for another user (e.g., teammate "zehuaw")
    python3 train.py --config.namespace zehuaw
    # Results appear in: zehuaw/ml-experiments/...

    # Example 3: Run with custom project name
    python3 train.py --config.namespace zehuaw --config.project my-research
    # Results appear in: zehuaw/my-research/...

    # Example 4: Override hyperparameters
    python3 train.py --config.learning-rate 0.001 --config.batch-size 64

    # Example 5: Combine namespace and hyperparameters
    python3 train.py \
        --config.namespace zehuaw \
        --config.learning-rate 0.01 \
        --config.batch-size 128 \
        --config.optimizer Adam

    # Example 6: From sweep launcher (run_sweep.py)
    python3 train.py --config.learning-rate 0.01 --config.batch-size 32 --config.sweep-index 0

Configuration:
    - Namespace: Auto-detected from authenticated user (via userinfo)
    - Override with: --config.namespace zehuaw (or any other namespace)
    - Project name: Set via --config.project (default: ml-experiments)
    - Full path: {namespace}/{project}/2026/01-26/experiments/sweeps/train/{job_counter:03d}

Expected path_stem: experiments/sweeps/train

The job_counter increments automatically for each sweep run, organizing all
sweep experiments under the same parent directory.
"""
import random
import time

from params_proto import proto

# Define hyperparameters using params-proto
# The @proto.prefix decorator creates a singleton config with namespaced CLI parsing
@proto.prefix
class Config:
    """Training configuration with params-proto.

    CLI args use format: --config.param-name value
    """

    # ML-Dash configuration
    namespace: str = None
    # Namespace (auto-detected from userinfo if not set)

    project: str = "ml-experiments"
    # Project name

    # Sweep parameters (set via CLI from sweep launcher)
    learning_rate: float = 0.01
    # Learning rate for optimizer

    batch_size: int = 32
    # Training batch size

    optimizer: str = "SGD"
    # Optimizer type: SGD or Adam

    momentum: float = 0.9
    # Momentum for SGD optimizer

    sweep_index: int = 0
    # Index of this run in the sweep

    # Fixed parameters
    model: str = "ResNet-18"
    # Model architecture

    dataset: str = "CIFAR-10"
    # Dataset name

    epochs: int = 5
    # Number of training epochs (5 for demo)

    weight_decay: float = 5e-4
    # Weight decay for regularization

    # Metadata
    sweep_id: str = "resnet_lr_batch_opt_sweep"
    # Unique identifier for this sweep

    total_sweep_runs: int = 8
    # Total number of runs in the sweep


@proto.cli
def main():
    """Run training with hyperparameter configuration.

    All Config parameters can be overridden via CLI:
        --config.learning-rate 0.001
        --config.batch-size 64
        --config.optimizer Adam
    """
    print("=" * 80)
    print("HYPERPARAMETER SWEEP: ResNet Training (params-proto)")
    print("=" * 80)

    print(f"\n📋 Hyperparameters for this run:")
    print(f"   Learning Rate: {Config.learning_rate}")
    print(f"   Batch Size:    {Config.batch_size}")
    print(f"   Optimizer:     {Config.optimizer}")
    print(f"   Momentum:      {Config.momentum}")
    print(f"   Sweep Index:   {Config.sweep_index}")

    # Configure ML-Dash BEFORE importing dxp
    # This must be done before dxp initialization
    import os
    from datetime import datetime
    from ml_dash.run import RUN
    from ml_dash import userinfo

    # Auto-detect namespace if not set
    namespace = Config.namespace
    if namespace is None:
        namespace = userinfo.username
        if namespace is None:
            raise ValueError(
                "Namespace not set and could not auto-detect from userinfo. "
                "Please set --config.namespace or authenticate with 'ml-dash login'"
            )

    # Construct full project path: {namespace}/{project}
    full_project = f"{namespace}/{Config.project}"
    RUN.project = full_project

    # CRITICAL: Set entry point for reliable auto-detection with `uv run`
    # Without this, path_stem may be detected as .venv/lib/.../ml_dash/experiment (wrong!)
    # With this, path_stem is correctly detected as experiments/sweeps/train
    RUN.entry = __file__

    # CRITICAL: For sweep runs, use shared timestamp and job counter from environment
    # This ensures all runs in a sweep are grouped under the same timestamp folder
    sweep_timestamp = os.environ.get("ML_DASH_SWEEP_TIMESTAMP")
    sweep_job_counter = os.environ.get("ML_DASH_JOB_COUNTER")

    if sweep_timestamp:
        # Convert Unix timestamp back to datetime
        RUN.now = datetime.fromtimestamp(float(sweep_timestamp))
        print(f"\n🕐 Using sweep timestamp: {RUN.now.strftime('%H.%M.%S')} (shared across all runs)")

    if sweep_job_counter:
        # Use the job counter from sweep launcher (1, 2, 3, ...)
        RUN.job_counter = int(sweep_job_counter)
        print(f"🔢 Using job counter: {RUN.job_counter:03d} (from sweep launcher)")

    print(f"\n🔧 ML-Dash Configuration:")
    print(f"   Namespace: {namespace}")
    print(f"   Project:   {Config.project}")
    print(f"   Full path: {full_project}")

    # Now import dxp (it will use the configured project, entry point, timestamp, and job counter)
    from ml_dash.auto_start import dxp

    print(f"\n🔍 Auto-Detection Results:")
    print(f"   Path stem: {dxp.run.path_stem}")
    print(f"   Full prefix: {dxp.run.prefix}")
    print(f"   Experiment name: {dxp.run.name}")
    print(f"   Job counter: {dxp.run.job_counter - 1}")

    print(f"\n📊 Dashboard Organization:")
    print(f"   All sweep runs organized under: experiments → sweeps → train")
    print(f"   This run: experiments/sweeps/train/{dxp.run.name}")

    print("\n" + "=" * 80)
    print(f"Starting Training (Sweep {Config.sweep_index + 1}/{Config.total_sweep_runs})...")
    print("=" * 80)

    # Run experiment with ML-Dash tracking
    with dxp.run:
        # Log which sweep configuration this is
        dxp.log(f"🔄 Hyperparameter Sweep Run {Config.sweep_index + 1}/{Config.total_sweep_runs}", level="info")
        dxp.log(f"Testing: lr={Config.learning_rate}, batch={Config.batch_size}, opt={Config.optimizer}", level="info")

        # ✨ params-proto advantage: One-line parameter logging!
        # Use Config._dict to get a clean dict (v3.x feature)
        dxp.params.set(**Config._dict)

        # Simulate training with these hyperparameters
        # Performance varies based on hyperparameters

        # Learning rate affects convergence speed
        lr_factor = {
            0.1: 0.85,      # Too high, unstable
            0.01: 1.0,      # Good
            0.001: 0.95,    # Slower convergence
            0.0001: 0.80,   # Very slow
        }.get(Config.learning_rate, 0.9)

        # Batch size affects generalization
        batch_factor = {
            32: 1.0,        # Better generalization
            64: 0.98,       # Good balance
            128: 0.95,      # Faster but less generalization
        }.get(Config.batch_size, 0.95)

        # Optimizer affects performance
        opt_factor = 1.0 if Config.optimizer == "Adam" else 0.97

        # Combined performance factor
        perf_factor = lr_factor * batch_factor * opt_factor

        best_val_acc = 0.0

        # Simulate training epochs
        for epoch in range(1, Config.epochs + 1):
            dxp.log(f"Epoch {epoch}/{Config.epochs}", level="info")

            # Training metrics (improves over time)
            train_loss = (2.5 * (0.65 ** epoch) / perf_factor) + random.uniform(0, 0.1)
            train_acc = min(0.98, (0.3 + epoch * 0.13) * perf_factor + random.uniform(0, 0.03))

            # Validation metrics
            val_loss = (2.3 * (0.65 ** epoch) / perf_factor) + random.uniform(0, 0.15)
            val_acc = min(0.95, (0.25 + epoch * 0.13) * perf_factor + random.uniform(0, 0.03))

            best_val_acc = max(best_val_acc, val_acc)

            # Log metrics
            dxp.metrics("train").log(
                epoch=epoch,
                loss=train_loss,
                accuracy=train_acc,
            )

            dxp.metrics("validation").log(
                epoch=epoch,
                loss=val_loss,
                accuracy=val_acc,
            )

            print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                  f"best_val={best_val_acc:.4f}")

            time.sleep(0.3)

        # Log final results
        dxp.log(f"✅ Training complete! Best validation accuracy: {best_val_acc:.4f}", level="info")

        # Log summary metrics for sweep comparison
        dxp.metrics("summary").log(
            best_val_accuracy=best_val_acc,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
        )

    print("\n" + "=" * 80)
    print("✓ Sweep Run Complete")
    print("=" * 80)
    print(f"\nThis run (#{Config.sweep_index + 1}): Best Val Accuracy = {best_val_acc:.4f}")
    print(f"\nView all sweep runs at:")
    print(f"  https://dash.ml/{dxp.run.owner}/{dxp.run.project}")
    print(f"  Navigate to: experiments → sweeps → train")
    print(f"\nAll {Config.total_sweep_runs} sweep configurations will be organized together!")
    print("=" * 80)


if __name__ == "__main__":
    main()
