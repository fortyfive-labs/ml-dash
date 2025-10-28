#!/usr/bin/env python3
"""Test script for parameters.read() and metrics.read() with remote backend"""

import sys
sys.path.insert(0, '/Users/57block/PycharmProjects/vuer-dashboard/ml-logger/src')

from ml_dash import Experiment

# Create experiment with remote backend
exp = Experiment(
    namespace="test",
    workspace="params-metrics-test",
    prefix="read-test",
    remote="http://localhost:4000"
)

print("Testing Parameters.read() and Metrics.read() with remote backend")
print("=" * 60)

# Start a run
with exp.run():
    # Set parameters
    print("\n1. Setting parameters...")
    exp.params.set(
        model="resnet50",
        learning_rate=0.001,
        batch_size=32,
        optimizer={
            "name": "adam",
            "beta1": 0.9,
            "beta2": 0.999
        }
    )

    # Extend parameters
    print("2. Extending parameters...")
    exp.params.extend(
        epochs=100,
        optimizer={"epsilon": 1e-8}  # This should merge with existing optimizer
    )

    # Update a single parameter
    print("3. Updating single parameter...")
    exp.params.update("learning_rate", 0.0001)

    # Read parameters back
    print("\n4. Reading parameters from server...")
    params = exp.params.read()

    print("\nFetched parameters:")
    print("-" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Verify the parameters
    assert params["model"] == "resnet50", "Model parameter mismatch"
    assert params["learning_rate"] == 0.0001, "Learning rate update failed"
    assert params["batch_size"] == 32, "Batch size parameter mismatch"
    assert params["epochs"] == 100, "Epochs parameter mismatch"
    assert params["optimizer"]["name"] == "adam", "Optimizer name mismatch"
    assert params["optimizer"]["epsilon"] == 1e-8, "Optimizer epsilon mismatch"

    print("\n✓ Parameters.read() test passed!")

    # Log metrics
    print("\n5. Logging metrics...")
    for step in range(5):
        exp.metrics.log(
            step=step,
            loss=0.5 - step * 0.05,
            accuracy=0.7 + step * 0.04,
            val_loss=0.6 - step * 0.04,
            val_accuracy=0.65 + step * 0.05
        )

    # Read metrics back
    print("\n6. Reading metrics from server...")
    metrics = exp.metrics.read()

    print(f"\nFetched {len(metrics)} metric entries:")
    print("-" * 60)
    for i, entry in enumerate(metrics):
        step = entry.get("step", "N/A")
        metric_values = entry.get("metrics", {})
        print(f"\nStep {step}:")
        for name, value in metric_values.items():
            print(f"  {name}: {value:.4f}")

    # Verify metrics count
    assert len(metrics) == 5, f"Expected 5 metric entries, got {len(metrics)}"

    # Verify metric values
    first_entry = metrics[0]
    assert "loss" in first_entry["metrics"], "Missing 'loss' metric"
    assert "accuracy" in first_entry["metrics"], "Missing 'accuracy' metric"
    assert first_entry.get("step") == 0, "First entry should be step 0"

    print("\n✓ Metrics.read() test passed!")

print("\n" + "=" * 60)
print("✓ All tests completed successfully!")
