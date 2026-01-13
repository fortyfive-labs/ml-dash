"""
Example: Using EXP.project_root for automatic prefix detection.

This example shows how to set up a project structure where experiment
prefixes are automatically derived from the script's file path.

Project Structure:
    my-project/
    ├── experiments/
    │   ├── __init__.py          # Sets EXP.project_root
    │   ├── vision/
    │   │   ├── resnet/
    │   │   │   └── train.py     # EXP.prefix = "vision/resnet"
    │   │   └── vit/
    │   │       └── train.py     # EXP.prefix = "vision/vit"
    │   └── nlp/
    │       └── bert/
    │           └── train.py     # EXP.prefix = "nlp/bert"
    └── ml-dash-data/            # Local storage root

Usage:
    1. Set EXP.project_root in experiments/__init__.py (one-time setup)
    2. Call EXP.__post_init__(entry=__file__) in each training script
    3. The prefix is automatically computed from the relative path
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_dash import RUN, Experiment


def demo_project_root():
  """Demonstrate EXP.project_root auto-detection."""
  print("=" * 60)
  print("EXP.project_root Example")
  print("=" * 60)

  # Simulate project structure
  # In a real project, this would be in experiments/__init__.py
  project_root = Path(__file__).parent.parent / "experiments"

  # Create simulated experiment directories
  exp_dir = project_root / "vision/resnet"
  exp_dir.mkdir(parents=True, exist_ok=True)
  train_script = exp_dir / "train.py"
  train_script.write_text("# Training script placeholder")

  print(f"\n1. Project root: {project_root}")
  print(f"   Train script: {train_script}")

  # Set project_root (normally done in experiments/__init__.py)
  RUN.project_root = str(project_root)
  print(f"\n2. Set EXP.project_root = '{RUN.project_root}'")

  # Call __post_init__ with the training script path
  # (normally this would be __file__ in the training script)
  RUN.__post_init__(entry=str(train_script))

  print(f"\n3. Called EXP.__post_init__(entry='{train_script}')")
  print("   Result:")
  print(f"   - EXP.prefix = '{RUN.prefix}'")
  print(f"   - EXP.name = '{RUN.name}'")
  print(f"   - EXP.entry = '{RUN.entry}'")

  # Now use with Experiment
  print("\n4. Creating experiment with auto-detected prefix:")

  data_dir = Path(__file__).parent / "tutorial_data"
  # Prefix format: owner/project/experiment-name
  with Experiment(
    prefix=f"demo/project/{RUN.prefix}",  # Uses auto-detected prefix
    local_path=str(data_dir),
    description="Demo using project_root",
  ).run as exp:
    exp.log("Training started!")
    exp.params.set(script=RUN.entry, model="resnet50", lr=0.001)
    print(f"   Experiment: {exp.project}/{exp.name}")
    print(f"   Data stored in: {exp._storage.root_path}")

  # Cleanup simulated directories
  import shutil

  if project_root.exists():
    shutil.rmtree(project_root)
  if data_dir.exists():
    shutil.rmtree(data_dir)

  print("\n" + "=" * 60)
  print("Summary:")
  print("  1. Set EXP.project_root in experiments/__init__.py")
  print("  2. Call EXP.__post_init__(entry=__file__) in train.py")
  print("  3. Use EXP.prefix for automatic path-based organization")
  print("=" * 60)


def demo_with_sweep_directory():
  """Demonstrate using a sweep.jsonl directory as entry."""
  print("\n" + "=" * 60)
  print("Using sweep.jsonl directory as entry")
  print("=" * 60)

  # Reset EXP for this demo
  RUN.prefix = None
  RUN.name = "scratch"
  RUN.entry = None

  # Simulate project structure with sweep files
  project_root = Path(__file__).parent.parent / "experiments"
  sweep_dir = project_root / "nlp/bert"
  sweep_dir.mkdir(parents=True, exist_ok=True)
  sweep_file = sweep_dir / "sweep.jsonl"
  sweep_file.write_text('{"lr": 0.001}\n{"lr": 0.0001}\n')

  print(f"\n1. Project root: {project_root}")
  print(f"   Sweep directory: {sweep_dir}")
  print(f"   Sweep file: {sweep_file}")

  # Set project_root
  RUN.project_root = str(project_root)

  # Use sweep directory as entry (instead of __file__)
  RUN.__post_init__(entry=str(sweep_dir))

  print(f"\n2. Called EXP.__post_init__(entry='{sweep_dir}')")
  print("   Result:")
  print(f"   - EXP.prefix = '{RUN.prefix}'")
  print(f"   - EXP.name = '{RUN.name}'")

  # Cleanup
  import shutil

  if project_root.exists():
    shutil.rmtree(project_root)

  print("\n" + "=" * 60)


if __name__ == "__main__":
  demo_project_root()
  demo_with_sweep_directory()
