"""
Baseline Sweep Launcher

Launches baseline training runs with STATIC paths (no datetime).
All baselines appear at consistent paths for permanent reference.

Usage:
    # Run default baseline sweep
    python launch_baseline.py

    # Run specific baseline sweep
    python launch_baseline.py --sweep configs/resnet_baseline.jsonl

    # Preview commands without running
    python launch_baseline.py --dry-run

Path structure (NO datetime):
    {namespace}/ml-experiments/baselines/resnet18/001
    {namespace}/ml-experiments/baselines/resnet18/002
    {namespace}/ml-experiments/baselines/resnet50/001
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from params_proto import proto
from ml_dash.run import RUN


@proto.cli
def main(
    sweep: str = "configs/resnet_baseline.jsonl",  # Sweep file path
    script: str = "train_baseline.py",  # Training script
    dry_run: bool = False,  # Show commands without running
):
    """Launch baseline sweep."""

    # Collect RUN arguments to forward to child processes
    run_args = []
    if RUN.owner:
        run_args.extend(["--run.owner", RUN.owner])
    if RUN.project and RUN.project != "{user}/scratch":  # Not default
        run_args.extend(["--run.project", RUN.project])

    # Resolve sweep file path
    sweep_path = Path(sweep)
    if not sweep_path.is_absolute():
        sweep_path = Path(__file__).parent / sweep

    if not sweep_path.exists():
        print(f"❌ Error: Sweep file not found: {sweep}")
        sys.exit(1)

    # Resolve training script
    train_script = Path(script)
    if not train_script.is_absolute():
        train_script = Path(__file__).parent / script

    if not train_script.exists():
        print(f"❌ Error: Training script not found: {script}")
        sys.exit(1)

    # Load sweep configurations
    configs = []
    with open(sweep_path, 'r') as f:
        for line in f:
            if line.strip():
                configs.append(json.loads(line))

    # Print launch info
    header = f"""
{'='*80}
BASELINE SWEEP LAUNCHER
{'='*80}

📂 Configuration:
   Sweep File:     {sweep_path.name}
   Train Script:   {train_script.name}
   Configurations: {len(configs)}

🔷 Baseline Mode: STATIC PATHS (no datetime)
   All runs use consistent paths for permanent reference

📋 Loaded Configurations:
"""
    print(header)

    # Print config table
    if configs:
        keys = list(configs[0].keys())
        header_row = f"   {'#':<4} " + " ".join(f"{k:<12}" for k in keys)
        print(header_row)
        print("   " + "-" * (len(header_row) - 3))

        for i, config in enumerate(configs, 1):
            values = " ".join(f"{str(config.get(k, '')):<12}" for k in keys)
            print(f"   {i:<4} {values}")

    print(f"\n{'='*80}")

    if dry_run:
        print("DRY RUN MODE - Commands will be shown but not executed")
        print(f"{'='*80}\n")

    # Track results
    results = []

    # Run training for each configuration
    for i, config in enumerate(configs):
        run_header = f"""
{'='*80}
BASELINE RUN {i+1}/{len(configs)}
{'='*80}
"""
        print(run_header)

        # Build command
        cmd = [sys.executable, str(train_script)]

        # Add sweep metadata
        cmd.extend(["--sweep-index", str(i)])
        if sweep_path.stem != "resnet_baseline":
            cmd.extend(["--sweep-id", sweep_path.stem])

        # Add RUN arguments (owner, project, etc.)
        cmd.extend(run_args)

        # Add config parameters with proper prefixes
        for key, value in config.items():
            # Determine prefix based on parameter name
            if key in ['learning_rate', 'batch_size', 'optimizer', 'momentum', 'epochs', 'weight_decay']:
                prefix = "train"
            elif key in ['name', 'model', 'dropout', 'pretrained']:
                prefix = "model"
            elif key in ['metric', 'dataset', 'test_batch_size']:
                prefix = "eval"
            else:
                prefix = None

            # Special handling for 'model' parameter (should map to Model.name)
            param_name = "name" if key == "model" else key.replace('_', '-')

            # Format CLI argument
            if prefix:
                cmd.extend([f"--{prefix}.{param_name}", str(value)])
            else:
                cmd.extend([f"--{param_name}", str(value)])

        # Set environment variable for job counter (no shared timestamp for baselines)
        env = os.environ.copy()
        env["ML_DASH_JOB_COUNTER"] = str(i + 1)

        if dry_run:
            print(f"Command: {' '.join(cmd)}")
            print(f"Env: ML_DASH_JOB_COUNTER={env['ML_DASH_JOB_COUNTER']}\n")
            results.append((i+1, config, "DRY RUN"))
            continue

        # Execute training
        try:
            result = subprocess.run(cmd, env=env, capture_output=False, text=True)

            if result.returncode == 0:
                status = "✓ SUCCESS"
            else:
                status = f"✗ FAILED (exit {result.returncode})"

            results.append((i+1, config, status))
            print(f"\n{status}\n")

        except Exception as e:
            status = f"✗ ERROR: {str(e)}"
            results.append((i+1, config, status))
            print(f"\n{status}\n")

    # Print summary
    summary = f"""
{'='*80}
BASELINE SWEEP SUMMARY
{'='*80}

Total Runs: {len(configs)}
Successes:  {sum(1 for _, _, s in results if '✓' in s)}
Failures:   {sum(1 for _, _, s in results if '✗' in s)}

Results:
"""
    print(summary)

    for run_num, config, status in results:
        config_str = ", ".join(f"{k}={v}" for k, v in list(config.items())[:3])
        if len(config) > 3:
            config_str += ", ..."
        print(f"  Run {run_num:2d}: {config_str:<50} {status}")

    footer = f"""
{'='*80}

Note: Baseline runs use STATIC paths and appear at consistent locations
      for permanent reference and easy comparison.
{'='*80}
"""
    print(footer)


if __name__ == "__main__":
    main()
