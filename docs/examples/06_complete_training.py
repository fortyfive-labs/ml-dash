"""Complete training example - Full end-to-end ML experiment metricing."""

import sys

sys.path.insert(0, "../../src")

import random
import time

from ml_dash import Experiment


def simulate_training_epoch():
  """Simulate one training epoch."""
  # Simulate training time
  time.sleep(0.2)

  # Simulate metrics
  train_loss = random.uniform(0.3, 0.5)
  train_acc = random.uniform(0.75, 0.85)
  val_loss = random.uniform(0.35, 0.55)
  val_acc = random.uniform(0.70, 0.80)

  return train_loss, train_acc, val_loss, val_acc


def main():
  print("=" * 60)
  print("Complete Training Example")
  print("Simulating full ML experiment with all ML-Dash features")
  print("=" * 60)

  # Training configuration
  config = {
    "model": {"architecture": "resnet50", "pretrained": True, "num_classes": 10},
    "training": {
      "optimizer": "adam",
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 10,
      "weight_decay": 0.0001,
    },
    "data": {
      "dataset": "cifar10",
      "augmentation": True,
      "train_split": 0.8,
      "val_split": 0.2,
    },
  }

  # Create ML-Dash experiment
  with Experiment(
    name="complete-training-demo",
    project="tutorials",
    local_path="./tutorial_data",
    description="Complete end-to-end training example",
    tags=["tutorial", "complete", "cifar10", "resnet"],
  ).run as experiment:
    # 1. Metric configuration
    print("\n[1/6] Metricing configuration...")
    experiment.params.set(**config)
    experiment.log("Configuration saved", level="info")

    # 2. Log training start
    print("[2/6] Starting training...")
    experiment.log(
      "Training started",
      level="info",
      metadata={
        "model": config["model"]["architecture"],
        "dataset": config["data"]["dataset"],
      },
    )

    best_val_acc = 0.0
    best_epoch = 0

    # 3. Training loop
    print("[3/6] Running training loop...")
    for epoch in range(config["training"]["epochs"]):
      # Simulate training
      train_loss, train_acc, val_loss, val_acc = simulate_training_epoch()

      # Make metrics improve over time
      train_loss = train_loss * (1 - epoch * 0.05)
      train_acc = min(0.95, train_acc + epoch * 0.01)
      val_loss = val_loss * (1 - epoch * 0.05)
      val_acc = min(0.92, val_acc + epoch * 0.01)

      # Metric metrics with epoch context
      lr = config["training"]["learning_rate"] * (0.95**epoch)
      experiment.metrics.log(epoch=epoch)
      experiment.metrics("train").log(loss=train_loss, accuracy=train_acc, lr=lr)
      experiment.metrics("eval").log(loss=val_loss, accuracy=val_acc)
      experiment.metrics.flush()

      # Log epoch summary
      experiment.log(
        f"Epoch {epoch + 1}/{config['training']['epochs']} complete",
        level="info",
        metadata={
          "epoch": epoch + 1,
          "train_loss": train_loss,
          "train_acc": train_acc,
          "val_loss": val_loss,
          "val_acc": val_acc,
          "lr": lr,
        },
      )

      print(
        f"   Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
      )

      # Save best model
      if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1

        # Create and upload "model" file
        import os

        os.makedirs("temp_models", exist_ok=True)
        model_path = "temp_models/best_model.txt"
        with open(model_path, "w") as f:
          f.write(f"Best model at epoch {best_epoch}\n")
          f.write(f"Validation accuracy: {best_val_acc:.4f}\n")

        experiment.files("models").upload(
          model_path,
          description=f"Best model (val_acc={best_val_acc:.4f})",
          tags=["best", "checkpoint"],
          metadata={"epoch": best_epoch, "val_accuracy": best_val_acc},
        )

        experiment.log(
          f"New best model saved (val_acc={best_val_acc:.4f})", level="info"
        )

    # 4. Save final model
    print("[4/6] Saving final model...")
    final_model_path = "temp_models/final_model.txt"
    with open(final_model_path, "w") as f:
      f.write(f"Final model after {config['training']['epochs']} epochs\n")
      f.write(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n")

    experiment.files("models").upload(
      final_model_path, description="Final model after all epochs", tags=["final"]
    )

    # 5. Save results summary
    print("[5/6] Saving results...")
    results_path = "temp_models/results.txt"
    with open(results_path, "w") as f:
      f.write("Training Results\n")
      f.write("=" * 50 + "\n")
      f.write(f"Model: {config['model']['architecture']}\n")
      f.write(f"Dataset: {config['data']['dataset']}\n")
      f.write(f"Epochs: {config['training']['epochs']}\n")
      f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
      f.write(f"Best epoch: {best_epoch}\n")

    experiment.files("results").upload(
      results_path, description="Training results summary", tags=["results", "summary"]
    )

    # Clean up temp files
    import shutil

    shutil.rmtree("temp_models")

    # 6. Final log
    print("[6/6] Finalizing...")
    experiment.log(
      "Training complete!",
      level="info",
      metadata={
        "total_epochs": config["training"]["epochs"],
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
      },
    )

  print("\n" + "=" * 60)
  print("Training Complete!")
  print("=" * 60)
  print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
  print("\nAll data saved to: tutorial_data/.dash/tutorials/complete-training-demo/")
  print("\nExplore your results:")
  print("  - Logs: cat tutorial_data/.dash/tutorials/complete-training-demo/logs.jsonl")
  print(
    "  - Parameters: cat tutorial_data/.dash/tutorials/complete-training-demo/parameters.json"
  )
  print("  - Metrics: ls tutorial_data/.dash/tutorials/complete-training-demo/metrics/")
  print("  - Files: ls tutorial_data/.dash/tutorials/complete-training-demo/files/")
  print("=" * 60)


if __name__ == "__main__":
  main()
