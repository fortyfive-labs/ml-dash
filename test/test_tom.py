import random
import time
from ml_dash import Experiment

def train_simple_model():
    """Train a simple model and metric everything with ML-Dash."""

    with Experiment(
        name="hyperparameter-schema-test-5",
        project="tutorials",
        folder='/tmp/ml_dash/examples',
        description="Comprehensive hyperparameter search across learning rate, batch size, and architecture",
        tags=["sweep", "best"],
        # remote='http://localhost:3000',
        # remote="https://api.dash.ml",
        # user_name="tom"
        local_path='.ml-dash'
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
            dropout=0.5
        )

        experiment.log("Starting training", level="info")

        # Training loop
        for epoch in range(50):
            epoch_start = time.time()

            # Learning rate decay
            current_lr = initial_lr * (0.95 ** epoch)

            # Simulate training metrics with realistic patterns
            train_loss = 2.5 * (0.85 ** epoch) + random.uniform(-0.05, 0.05)
            val_loss = 2.6 * (0.87 ** epoch) + random.uniform(-0.08, 0.08)

            # Classification metrics
            train_accuracy = min(0.99, 0.3 + epoch * 0.012) + random.uniform(-0.01, 0.01)
            val_accuracy = min(0.96, 0.28 + epoch * 0.011) + random.uniform(-0.02, 0.02)

            # Precision, Recall, F1
            precision = min(0.97, 0.35 + epoch * 0.011) + random.uniform(-0.015, 0.015)
            recall = min(0.96, 0.32 + epoch * 0.012) + random.uniform(-0.015, 0.015)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Advanced metrics
            gradient_norm = 5.0 / (epoch + 1) + random.uniform(-0.2, 0.2)
            perplexity = 2 ** val_loss

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
            experiment.metrics("train").append(
                loss=train_loss,
                accuracy=train_accuracy,
                epoch=epoch
            )

            # Validation metrics
            experiment.metrics("validation").append(
                loss=val_loss,
                accuracy=val_accuracy,
                perplexity=perplexity,
                epoch=epoch
            )

            # Classification metrics
            experiment.metrics("classification").append(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                true_positives=true_positives,
                false_positives=false_positives,
                epoch=epoch
            )

            # Per-class accuracy
            experiment.metrics("per_class").append(
                class_0_accuracy=class_0_acc,
                class_1_accuracy=class_1_acc,
                class_2_accuracy=class_2_acc,
                epoch=epoch
            )

            # Optimization metrics
            experiment.metrics("optimization").append(
                learning_rate=current_lr,
                gradient_norm=gradient_norm,
                epoch=epoch
            )

            # Resource metrics
            experiment.metrics("resources").append(
                gpu_memory_mb=gpu_memory_mb,
                batch_time_ms=batch_time_ms,
                epoch_time_sec=epoch_time,
                epoch=epoch
            )

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
                        "learning_rate": current_lr
                    }
                )

        # Save model file
        # In real code: torch.save(model.state_dict(), "model.pth")
        with open("model.pth", "w") as f:
            f.write("model weights")

        experiment.files(
            file_path="model.pth",
            prefix="/models",
            description="Final trained model",
            tags=["final"]
        ).save()
        experiment.files(
            file_path="/Users/57block/fortyfive/ml-dash/test/selfie.jpeg",
            prefix="/plots",
            description="Final trained model",
            tags=["final"]
        ).save()  
        experiment.files(
            file_path="/Users/57block/fortyfive/ml-dash/test/view.yaml",
            description="Final trained model",
            tags=["final"]
        ).save()     
        experiment.files(
            file_path="/Users/57block/fortyfive/ml-dash/test/video.mp4",
            prefix="/videos",
            description="Final trained model",
            tags=["final"]
        ).save()  
        experiment.files(
            file_path="./README.md",
            description="Best model checkpoint",
            tags=["best", "checkpoint"],
            metadata={"epoch": 50, "val_accuracy": 0.96, "f1_score": 0.96}
        ).save()

        experiment.log("Training complete!", level="info")
        print(f"✓ Experiment completed successfully with 18 metrics tracked")

if __name__ == "__main__":
    train_simple_model()


# The current code incorrectly uses metric
#    names as namespaces. The correct approach is to group related metrics
#   in namespaces and pass multiple metrics per append call


            # # Log all metrics (15+ metrics total)
            # experiment.metrics("train_loss").append(value=train_loss, epoch=epoch)
            # experiment.metrics("val_loss").append(value=val_loss, epoch=epoch)
            # experiment.metrics("train_accuracy").append(value=train_accuracy, epoch=epoch)
            # experiment.metrics("val_accuracy").append(value=val_accuracy, epoch=epoch)
            # experiment.metrics("precision").append(value=precision, epoch=epoch)
            # experiment.metrics("recall").append(value=recall, epoch=epoch)
            # experiment.metrics("f1_score").append(value=f1_score, epoch=epoch)
            # experiment.metrics("learning_rate").append(value=current_lr, epoch=epoch)
            # experiment.metrics("gradient_norm").append(value=gradient_norm, epoch=epoch)
            # experiment.metrics("perplexity").append(value=perplexity, epoch=epoch)
            # experiment.metrics("gpu_memory_mb").append(value=gpu_memory_mb, epoch=epoch)
            # experiment.metrics("batch_time_ms").append(value=batch_time_ms, epoch=epoch)
            # experiment.metrics("epoch_time_sec").append(value=epoch_time, epoch=epoch)
            # experiment.metrics("true_positives").append(value=true_positives, epoch=epoch)
            # experiment.metrics("false_positives").append(value=false_positives, epoch=epoch)
            # experiment.metrics("class_0_accuracy").append(value=class_0_acc, epoch=epoch)
            # experiment.metrics("class_1_accuracy").append(value=class_1_acc, epoch=epoch)
            # experiment.metrics("class_2_accuracy").append(value=class_2_acc, epoch=epoch)