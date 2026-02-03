import getpass
from math import exp
import random
import time

from ml_dash import Experiment


def train_simple_model():
  """Train a simple model and metric everything with ML-Dash."""

  # Get current user for prefix
  owner = getpass.getuser()

  with Experiment(
    prefix=f"tom/summary-project-3/f/s-test-0",
    readme="Comprehensive hyperparameter search across learning rate, batch size, and architecture",
    tags=["sweep", "best"],
    dash_url='http://localhost:3000',  # Use for local server testing
    # dash_url="https://api.dash.ml",  # Use for remote mode
    # dash_root=".dash",  # Local storage directory
  ).run as experiment:
    # Metric hyperparameters
    initial_lr = 0.001
    experiment.params.set(
      learning_rate=initial_lr,
      batch_size=32,
      epochs=50,
      optimizer="adam",
      model="resnet50",
      weight_decay=0.0001,
      momentum=0.9,
      dropout=0.5,
    )

    experiment.log("Starting training", level="info")

    # Training loop
    for epoch in range(100000):
      epoch_start = time.time()

      # Learning rate decay
      current_lr = initial_lr * (0.95**epoch)

      # Simulate training metrics with realistic patterns
      train_loss = 2.5 * (0.85**epoch) + random.uniform(-0.05, 0.05)
      val_loss = 2.6 * (0.87**epoch) + random.uniform(-0.08, 0.08)

      # Classification metrics
      train_accuracy = min(0.99, 0.3 + epoch * 0.012) + random.uniform(-0.01, 0.01)
      val_accuracy = min(0.96, 0.28 + epoch * 0.011) + random.uniform(-0.02, 0.02)

      # Precision, Recall, F1
      precision = min(0.97, 0.35 + epoch * 0.011) + random.uniform(-0.015, 0.015)
      recall = min(0.96, 0.32 + epoch * 0.012) + random.uniform(-0.015, 0.015)
      f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

      # Advanced metrics
      gradient_norm = 5.0 / (epoch + 1) + random.uniform(-0.2, 0.2)
      perplexity = 2**val_loss

      # Resource metrics
      gpu_memory_mb = 2048 + random.uniform(-50, 100)
      batch_time_ms = 120 + random.uniform(-10, 15)

      # Confusion matrix metrics
      true_positives = int(900 + epoch * 2 + random.uniform(-20, 20))
      false_positives = int(100 - epoch * 1.5 + random.uniform(-10, 10))

      # Per-class accuracy (simulating multi-class)
      class_0_acc = min(0.98, 0.4 + epoch * 0.01) + random.uniform(-0.02, 0.02)
      class_1_acc = min(0.97, 0.35 + epoch * 0.012) + random.uniform(-0.02, 0.02)
      class_2_acc = min(0.95, 0.32 + epoch * 0.011) + random.uniform(-0.02, 0.02)

      epoch_time = time.time() - epoch_start

      # Log all metrics (18 metrics total, grouped by category)
      # Training metrics
    #   experiment.metrics("train").log(
    #     loss=train_loss, accuracy=train_accuracy, epoch=epoch
    #   )

      experiment.metrics("validation").log(
        epoch=epoch, loss=val_loss, accuracy=val_accuracy, perplexity=perplexity
      )
    #   experiment.metrics.buffer.log_summary()
    #   experiment.metrics.flush()  

     

      # Log progress every 10 epochs
      if (epoch + 1) % 10 == 0:
        experiment.log(
          f"Epoch {epoch + 1}/50 complete",
          level="info",
          metadata={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "f1_score": f1_score,
            "learning_rate": current_lr,
          },
        )

    # # Save model file
    # # In real code: torch.save(model.state_dict(), "model.pth")
    # with open("model.pth", "w") as f:
    #   f.write("model weights")

    # # Upload model file
    # experiment.files("models").upload(
    #   "model.pth", description="Final trained model", tags=["final"]
    # )

    # # Upload example image
    # experiment.files("plots").upload(
    #   "/Users/57block/fortyfive/ml-dash/test/selfie.jpeg",
    #   description="Training visualization",
    #   tags=["plot", "final"],
    # )

    # # Upload configuration file
    # experiment.files().upload(
    #   "/Users/57block/fortyfive/ml-dash/test/.dashrc",
    #   description="Configuration file",
    #   tags=["config"],
    # )

    # # Upload example video
    # experiment.files("videos").upload(
    #   "/Users/57block/fortyfive/ml-dash/test/video.mp4",
    #   description="Training progress video",
    #   tags=["video", "final"],
    # )

    # # Upload README with metadata
    # experiment.files().upload(
    #   "./README.md",
    #   description="Best model checkpoint",
    #   tags=["best", "checkpoint"],
    #   metadata={"epoch": 50, "val_accuracy": 0.96, "f1_score": 0.96},
    # )


    experiment.log("Training complete!", level="info")
    print("✓ Experiment completed successfully with 18 metrics tracked")


if __name__ == "__main__":
  train_simple_model()
