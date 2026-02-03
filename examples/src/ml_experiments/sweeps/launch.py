"""
Hyperparameter Sweep Launcher

Launches multiple training runs with configurations from a sweep file.
All runs share the same timestamp for organized dashboard grouping.

Usage:
    # Run default sweep
    python launch.py

    # Run specific sweep file
    python launch.py --sweep configs/lr_sweep.jsonl

    # Override RUN settings
    python launch.py --owner zehuaw --project my-research

    # Use local API server
    python launch.py --api-url http://localhost:3000

    # Dry run
    python launch.py --sweep configs/optimizer_sweep.jsonl --dry-run
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from params_proto import proto


@proto.cli
def main(
    sweep: str = "configs/sweep.jsonl",  # Sweep file path
    script: str = "train.py",  # Training script
    dry_run: bool = False,  # Show commands without running
    owner: str = None,  # ML-Dash owner/namespace
    project: str = None,  # ML-Dash project name
    api_url: str = None,  # ML-Dash API URL
):
    """Launch hyperparameter sweep."""

    # Collect RUN arguments to forward to child processes
    run_args = []
    if owner:
        run_args.extend(["--run.owner", owner])
    if project:
        run_args.extend(["--run.project", project])
    if api_url:
        run_args.extend(["--run.api-url", api_url])

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

    # Capture shared timestamp for all runs
    sweep_timestamp = datetime.now()
    sweep_timestamp_str = sweep_timestamp.strftime("%H.%M.%S")

    # Print launch info using template string
    header = f"""
{'='*80}
SWEEP LAUNCHER
{'='*80}

📂 Configuration:
   Sweep File:    {sweep_path.name}
   Train Script:  {train_script.name}
   Configurations: {len(configs)}

🕐 Sweep Timestamp: {sweep_timestamp_str} ({sweep_timestamp.strftime('%I:%M:%S %p')})
   All runs grouped under: .../{{timestamp}}/001, 002, ...

📋 Loaded Configurations:
"""
    print(header)

    # Print config table
    if configs:
        # Determine columns from first config
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
RUN {i+1}/{len(configs)}
{'='*80}
"""
        print(run_header)

        # Build command
        cmd = [sys.executable, str(train_script)]

        # Add sweep metadata
        cmd.extend(["--sweep-index", str(i)])
        if sweep_path.stem != "sweep":
            cmd.extend(["--sweep-id", sweep_path.stem])

        # Forward RUN arguments (--RUN.owner, --RUN.project)
        cmd.extend(run_args)

        # Add config parameters (use proper prefixes)
        for key, value in config.items():
            # Determine prefix based on parameter name
            if key in ['learning_rate', 'batch_size', 'optimizer', 'momentum', 'epochs', 'weight_decay']:
                prefix = "train"
            elif key in ['name', 'dropout', 'pretrained']:
                prefix = "model"
            elif key in ['metric', 'dataset', 'test_batch_size']:
                prefix = "eval"
            else:
                prefix = None

            # Format CLI argument
            param_name = key.replace('_', '-')
            if prefix:
                cmd.extend([f"--{prefix}.{param_name}", str(value)])
            else:
                cmd.extend([f"--{param_name}", str(value)])

        # Set environment variables for sweep coordination
        env = os.environ.copy()
        env["ML_DASH_SWEEP_TIMESTAMP"] = str(sweep_timestamp.timestamp())
        env["ML_DASH_JOB_COUNTER"] = str(i + 1)

        if dry_run:
            print(f"Command: {' '.join(cmd)}")
            print(f"Env: ML_DASH_SWEEP_TIMESTAMP={env['ML_DASH_SWEEP_TIMESTAMP']}")
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
SWEEP SUMMARY
{'='*80}

Total Runs: {len(configs)}
Successes:  {sum(1 for _, _, s in results if '✓' in s)}
Failures:   {sum(1 for _, _, s in results if '✗' in s)}

Results:
"""
    print(summary)

    for run_num, config, status in results:
        # Format config summary
        config_str = ", ".join(f"{k}={v}" for k, v in list(config.items())[:3])
        if len(config) > 3:
            config_str += ", ..."
        print(f"  Run {run_num:2d}: {config_str:<50} {status}")

    footer = f"""
{'='*80}
"""
    print(footer)


if __name__ == "__main__":
    main()
