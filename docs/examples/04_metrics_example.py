"""
Metrics Example - Time-series data tracking

This example demonstrates metric tracking with ML-Dash in local mode (no authentication required).

Usage:
    python 04_metrics_example.py
"""

import sys

sys.path.insert(0, "../../src")

import random

from ml_dash import Experiment


def main():
  print("=" * 60)
  print("Metrics Example - Time-Series Metrics")
  print("=" * 60)

  with Experiment(
    prefix="metrics-demo", project="tutorials"
  ).run as experiment:
    experiment.params.set(epochs=10, learning_rate=0.001)

    print("\n1. Logging training metrics...")

    # Log metrics over epochs
    for epoch in range(10):
      # Simulate training
      train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
      val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
      accuracy = min(0.95, 0.5 + epoch * 0.05)

      # Log metrics with epoch context
      experiment.metrics("train").log(loss=train_loss, accuracy=accuracy)
      experiment.metrics("eval").log(loss=val_loss, accuracy=accuracy)
      experiment.metrics.log(epoch=epoch).flush()

      print(f"   Epoch {epoch + 1}: loss={train_loss:.4f}, acc={accuracy:.4f}")

    print("\n2. Logging multiple data points...")

    # Log multiple points individually
    for i, (loss, step) in enumerate([(0.45, 100), (0.42, 200), (0.40, 300), (0.38, 400)]):
      experiment.metrics("batch").log(loss=loss, step=step, batch=i + 1)
    print(f"   Logged 4 data points")

    print("\n3. Flexible schema - multiple metrics per point...")

    # Log multiple metrics in one call
    experiment.metrics("combined").log(
      epoch=5,
      train_loss=0.3,
      val_loss=0.35,
      train_acc=0.85,
      val_acc=0.82,
      learning_rate=0.001,
    )

    print("\n4. Reading metric data...")

    # Read metric data
    result = experiment.metrics("train").read(start_index=0, limit=5)
    print(f"   Read {result['total']} data points:")
    for point in result["data"][:3]:
      print(f"     Index {point['index']}: {point['data']}")

    print("\n5. Getting metric statistics...")

    # Get metric stats
    stats = experiment.metrics("train").stats()
    print(f"   Metric: {stats['name']}")
    print(f"   Total points: {stats['totalDataPoints']}")

    print("\n6. Listing all metrics...")

    # List all metrics
    metrics = experiment.metrics("train").list_all()
    print(f"   Found {len(metrics)} metrics:")
    for metric in metrics:
      print(f"     - {metric['name']}: {metric['totalDataPoints']} points")

    experiment.log("Metrics demo complete", level="info")

  print("\n✓ All metrics logged!")
  print("\n" + "=" * 60)
  print("View metric data:")
  print(
    "  cat tutorial_data/.dash/tutorials/metrics-demo/metrics/train/data.jsonl"
  )
  print(
    "  cat tutorial_data/.dash/tutorials/metrics-demo/metrics/train/metadata.json"
  )
  print("=" * 60)


if __name__ == "__main__":
  main()
