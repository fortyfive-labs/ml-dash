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
    name="metrics-demo", project="tutorials", local_path="./tutorial_data"
  ).run as experiment:
    experiment.params.set(epochs=10, learning_rate=0.001)

    print("\n1. Metricing training metrics...")

    # Metric metrics over epochs
    for epoch in range(10):
      # Simulate training
      train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
      val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
      accuracy = min(0.95, 0.5 + epoch * 0.05)

      # Log metrics with epoch context
      experiment.metrics.log(epoch=epoch)
      experiment.metrics("train").log(loss=train_loss, accuracy=accuracy)
      experiment.metrics("eval").log(loss=val_loss, accuracy=accuracy)
      experiment.metrics.flush()

      print(f"   Epoch {epoch + 1}: loss={train_loss:.4f}, acc={accuracy:.4f}")

    print("\n2. Batch appending data points...")

    # Batch append for better performance
    batch_data = [
      {"value": 0.45, "step": 100, "batch": 1},
      {"value": 0.42, "step": 200, "batch": 2},
      {"value": 0.40, "step": 300, "batch": 3},
      {"value": 0.38, "step": 400, "batch": 4},
    ]
    result = experiment.metrics("step_loss").append_batch(batch_data)
    print(f"   Appended {result['count']} data points")

    print("\n3. Flexible schema - multiple metrics per point...")

    # Metric multiple metrics in one metric
    experiment.metrics("all_metrics").append(
      epoch=5,
      train_loss=0.3,
      val_loss=0.35,
      train_acc=0.85,
      val_acc=0.82,
      learning_rate=0.001,
    )

    print("\n4. Reading metric data...")

    # Read metric data
    result = experiment.metrics("train_loss").read(start_index=0, limit=5)
    print(f"   Read {result['total']} data points:")
    for point in result["data"][:3]:
      print(f"     Index {point['index']}: {point['data']}")

    print("\n5. Getting metric statistics...")

    # Get metric stats
    stats = experiment.metrics("train_loss").stats()
    print(f"   Metric: {stats['name']}")
    print(f"   Total points: {stats['totalDataPoints']}")

    print("\n6. Listing all metrics...")

    # List all metrics
    metrics = experiment.metrics("train_loss").list_all()
    print(f"   Found {len(metrics)} metrics:")
    for metric in metrics:
      print(f"     - {metric['name']}: {metric['totalDataPoints']} points")

    experiment.log("Metrics demo complete", level="info")

  print("\n✓ All metrics metriced!")
  print("\n" + "=" * 60)
  print("View metric data:")
  print(
    "  cat tutorial_data/.dash/tutorials/metrics-demo/metrics/train_loss/data.jsonl"
  )
  print(
    "  cat tutorial_data/.dash/tutorials/metrics-demo/metrics/train_loss/metadata.json"
  )
  print("=" * 60)


if __name__ == "__main__":
  main()
