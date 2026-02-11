"""
Demo: Organizing experiments with run.folder

Shows how to organize dxp/rdxp experiments into folders before initialization.
"""

from ml_dash import dxp
from datetime import datetime

print("Demo: Organizing Experiments with run.folder")
print("=" * 60)

# Example 1: Organize by date
print("\n1. Organizing by date:")
today = datetime.now().strftime('%Y-%m-%d')
dxp.run.folder = f"daily-runs/{today}"
print(f"   Set folder: {dxp.run.folder}")

with dxp.run:
    dxp.params.set(date=today, experiment_type="daily")
    print(f"   ✓ Experiment running in: {dxp.folder}")

# Example 2: Organize by project/model
print("\n2. Organizing by project and model:")
dxp.run.folder = "vision/classification/resnet50"

with dxp.run:
    dxp.params.set(
        model="resnet50",
        dataset="imagenet",
        task="classification"
    )
    print(f"   ✓ Experiment running in: {dxp.folder}")

# Example 3: Organize by hyperparameters
print("\n3. Organizing by hyperparameters:")
lr = 0.001
batch_size = 32
dxp.run.folder = f"experiments/lr_{lr}/batch_{batch_size}"

with dxp.run:
    dxp.params.set(learning_rate=lr, batch_size=batch_size)
    print(f"   ✓ Experiment running in: {dxp.folder}")

print("\n" + "=" * 60)
print("Benefits:")
print("  • Keep experiments organized in logical folders")
print("  • Easy to browse by date, model, or hyperparameters")
print("  • Folder is set before initialization (clean API)")
print("  • Prevents accidental folder changes during runtime")
print("=" * 60)
