"""
Hyperparameter Sweep Launcher

This script reads hyperparameter configurations from sweep.jsonl and launches
multiple training runs. Each run creates a separate experiment in ML-Dash,
organized under the same parent directory using job_counter.

Dashboard Organization:
  experiments/sweeps/train/
    ├── 001/  ← lr=0.1, batch=32, opt=SGD
    ├── 002/  ← lr=0.01, batch=32, opt=SGD
    ├── 003/  ← lr=0.001, batch=32, opt=SGD
    ├── 004/  ← lr=0.01, batch=64, opt=SGD
    ├── 005/  ← lr=0.01, batch=128, opt=SGD
    ├── 006/  ← lr=0.01, batch=32, opt=Adam
    ├── 007/  ← lr=0.001, batch=64, opt=Adam
    └── 008/  ← lr=0.0001, batch=128, opt=Adam

This makes it easy to compare all sweep runs side-by-side!
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run hyperparameter sweep with multiple configurations"
)
parser.add_argument(
    "--namespace",
    type=str,
    default=None,
    help="Override namespace for all sweep runs (e.g., zehuaw). Default: auto-detect from authenticated user"
)
parser.add_argument(
    "--project",
    type=str,
    default=None,
    help="Override project name for all sweep runs (default: ml-experiments)"
)
args = parser.parse_args()

print("=" * 80)
print("HYPERPARAMETER SWEEP LAUNCHER")
print("=" * 80)

if args.namespace:
    print(f"\n🔧 Namespace override: {args.namespace}")
    print(f"   All experiments will run under: {args.namespace}/...")
else:
    print(f"\n🔧 Namespace: Auto-detected from authenticated user")

if args.project:
    print(f"   Project: {args.project}")
else:
    print(f"   Project: ml-experiments (default)")

# Get the directory containing this script
sweep_dir = Path(__file__).parent
sweep_file = sweep_dir / "sweep.jsonl"
train_script = sweep_dir / "train.py"

print(f"\nSweep Configuration:")
print(f"  Directory:     {sweep_dir}")
print(f"  Sweep file:    {sweep_file.name}")
print(f"  Train script:  {train_script.name}")

# Load sweep configurations
configs = []
with open(sweep_file, 'r') as f:
    for line in f:
        if line.strip():
            configs.append(json.loads(line))

print(f"\n📋 Loaded {len(configs)} configurations from sweep.jsonl:")
print(f"{'#':<4} {'LR':<10} {'Batch':<8} {'Optimizer':<10} {'Momentum':<10}")
print("-" * 80)
for i, config in enumerate(configs, 1):
    print(f"{i:<4} {config['learning_rate']:<10} {config['batch_size']:<8} "
          f"{config['optimizer']:<10} {config.get('momentum', 0.0):<10}")

print("\n" + "=" * 80)
print("Starting Sweep Runs...")
print("=" * 80)

# CRITICAL: Capture timestamp ONCE for the entire sweep
# This ensures all runs share the same timestamp folder
sweep_timestamp = datetime.now()
sweep_timestamp_str = sweep_timestamp.strftime("%H.%M.%S")
print(f"\n🕐 Sweep timestamp: {sweep_timestamp_str} ({sweep_timestamp.strftime('%I:%M:%S %p')})")
print(f"   All runs will be grouped under: .../train/{sweep_timestamp_str}/")

# Track results
results = []

# Run training for each configuration
for i, config in enumerate(configs):
    print(f"\n{'=' * 80}")
    print(f"[{i+1}/{len(configs)}] Running Configuration #{i+1}")
    print("=" * 80)
    print(f"Parameters: lr={config['learning_rate']}, batch={config['batch_size']}, "
          f"opt={config['optimizer']}")
    print("-" * 80)

    # Build command with params-proto namespaced args
    # Format: --config.param-name (lowercase, kebab-case)
    cmd = [
        sys.executable,
        str(train_script),
        "--config.learning-rate", str(config['learning_rate']),
        "--config.batch-size", str(config['batch_size']),
        "--config.optimizer", config['optimizer'],
        "--config.momentum", str(config.get('momentum', 0.9)),
        "--config.sweep-index", str(i),
    ]

    # Add namespace override if provided
    if args.namespace:
        cmd.extend(["--config.namespace", args.namespace])

    # Add project override if provided
    if args.project:
        cmd.extend(["--config.project", args.project])

    # CRITICAL: Set environment variables to share timestamp and job counter
    # This ensures all runs are grouped under the same timestamp folder
    env = os.environ.copy()
    env["ML_DASH_SWEEP_TIMESTAMP"] = str(sweep_timestamp.timestamp())  # Unix timestamp
    env["ML_DASH_JOB_COUNTER"] = str(i + 1)  # Job counter starts at 1

    try:
        # Run the training script with shared timestamp
        result = subprocess.run(
            cmd,
            cwd=sweep_dir,
            capture_output=False,  # Show output in real-time
            text=True,
            env=env,  # Pass environment variables
        )

        if result.returncode == 0:
            status = "✓ SUCCESS"
            results.append((i+1, config, status))
        else:
            status = f"✗ FAILED (exit code {result.returncode})"
            results.append((i+1, config, status))

        print(f"\n{status}")

    except Exception as e:
        status = f"✗ ERROR: {str(e)}"
        results.append((i+1, config, status))
        print(f"\n{status}")

    print("-" * 80)

# Print summary
print("\n" + "=" * 80)
print("SWEEP SUMMARY")
print("=" * 80)

print(f"\n{'Run':<6} {'LR':<10} {'Batch':<8} {'Optimizer':<10} {'Status':<20}")
print("-" * 80)
for run_num, config, status in results:
    print(f"{run_num:<6} {config['learning_rate']:<10} {config['batch_size']:<8} "
          f"{config['optimizer']:<10} {status:<20}")

# Count successes
successes = sum(1 for _, _, status in results if "✓" in status)
print(f"\n{successes}/{len(configs)} runs completed successfully")

# Show dashboard information
print("\n" + "=" * 80)
print("VIEW RESULTS IN DASHBOARD")
print("=" * 80)
print("\nAll sweep experiments are organized together:")
print("  https://dash.ml/tom_tao_e4c2c9/ml-experiments/")
print("\nNavigate to:")
print("  experiments → sweeps → train")
print("\nYou'll see all 8 runs organized numerically:")
print("""
  train/
    ├── 001/  ← lr=0.1, batch=32, opt=SGD
    ├── 002/  ← lr=0.01, batch=32, opt=SGD
    ├── 003/  ← lr=0.001, batch=32, opt=SGD
    ├── 004/  ← lr=0.01, batch=64, opt=SGD
    ├── 005/  ← lr=0.01, batch=128, opt=SGD
    ├── 006/  ← lr=0.01, batch=32, opt=Adam
    ├── 007/  ← lr=0.001, batch=64, opt=Adam
    └── 008/  ← lr=0.0001, batch=128, opt=Adam
""")

print("\n💡 Tips for Analyzing Sweep Results:")
print("  1. Compare validation accuracy across runs")
print("  2. Look for best hyperparameter combinations")
print("  3. Analyze learning curves side-by-side")
print("  4. Check which optimizer performs better")
print("  5. Find optimal learning rate for each batch size")

print("\n" + "=" * 80)
