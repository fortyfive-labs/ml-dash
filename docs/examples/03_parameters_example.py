"""
Parameters Example - Track hyperparameters and configuration

This example demonstrates parameter tracking with ML-Dash in local mode (no authentication required).

Usage:
    python 03_parameters_example.py
"""

import sys

sys.path.insert(0, "../../src")

from ml_dash import Experiment


def main():
  print("=" * 60)
  print("Parameters Example")
  print("=" * 60)

  with Experiment(
    prefix="parameters-demo", project="tutorials"
  ).run as experiment:
    # Simple parameters
    experiment.params.set(learning_rate=0.001, batch_size=32, epochs=100)

    print("\n1. Metriced simple parameters")

    # Nested parameters (automatically flattened)
    experiment.params.set(
      **{
        "model": {"architecture": "resnet50", "pretrained": True, "num_layers": 50},
        "optimizer": {
          "type": "adam",
          "beta1": 0.9,
          "beta2": 0.999,
          "weight_decay": 0.0001,
        },
        "data": {"dataset": "imagenet", "augmentation": True, "num_workers": 4},
      }
    )

    print("2. Metriced nested parameters (auto-flattened)")

    # Update existing parameter
    experiment.params.set(learning_rate=0.0001)

    print("3. Updated learning_rate")

    # Add more parameters
    experiment.params.set(use_mixed_precision=True, gradient_clipping=1.0)

    print("4. Added additional parameters")

    experiment.log("Parameters configured", level="info")

  print("\n✓ All parameters saved!")
  print("\n" + "=" * 60)
  print("View parameters:")
  print("  cat tutorial_data/.dash/tutorials/parameters-demo/parameters.json")
  print("\nExpected flat structure:")
  print("  learning_rate: 0.0001 (updated)")
  print("  model.architecture: resnet50")
  print("  model.pretrained: true")
  print("  optimizer.type: adam")
  print("  ...")
  print("=" * 60)


if __name__ == "__main__":
  main()
