"""
Logging Example - Structured logging with different levels

This example demonstrates logging with ML-Dash in local mode (no authentication required).

Usage:
    python 02_logging_example.py
"""

import sys

sys.path.insert(0, "../../src")

import time

from ml_dash import Experiment


def main():
  print("=" * 60)
  print("Logging Example")
  print("=" * 60)

  with Experiment(
    prefix="logging-demo", project="tutorials", local_path="./tutorial_data"
  ).run as experiment:
    # Different log levels //
    experiment.log("Debug information", level="debug")
    experiment.log("Training started", level="info")
    experiment.log("GPU memory usage high", level="warn")
    experiment.log("Failed to load checkpoint", level="error")

    print("\n1. Testing different log levels...")

    # Log with metadata
    experiment.log(
      "Epoch completed",
      level="info",
      metadata={
        "epoch": 5,
        "train_loss": 0.234,
        "val_loss": 0.456,
        "learning_rate": 0.001,
      },
    )

    print("2. Logging with structured metadata...")

    # Simulate progress logging
    total = 100
    for i in range(0, total + 1, 10):
      percent = i
      experiment.log(
        f"Progress: {percent}%",
        level="info",
        metadata={"processed": i, "total": total, "percent": percent},
      )
      time.sleep(0.1)

    print("3. Progress logging complete...")

    # Error logging
    try:
      raise ValueError("Simulated error")
    except Exception as e:
      experiment.log(
        f"Error occurred: {str(e)}",
        level="error",
        metadata={"error_type": type(e).__name__, "error_message": str(e)},
      )

    print("4. Error logging complete...")

    experiment.log("Logging demo complete!", level="info")

  print("\n✓ All logs saved!")
  print("\n" + "=" * 60)
  print("View logs:")
  print("  cat tutorial_data/.dash/tutorials/logging-demo/logs.jsonl")
  print("=" * 60)


if __name__ == "__main__":
  main()
