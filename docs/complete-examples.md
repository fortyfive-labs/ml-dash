# Complete Examples

This document contains complete, runnable examples showcasing common use cases for ML-Dash.

## Example 1: Simple Training Loop

A basic training loop with logging, parameters, and metrics tracking.

```python
"""Simple training loop example."""
import random
from ml_dash import Experiment

def train_simple_model():
    """Train a simple model and metric with ML-Dash."""

    with Experiment(
        name="simple-training",
        project="tutorials",
        description="Simple training example",
        tags=["tutorial", "simple"],
        local_path=".dash"
    ).run as experiment:
        # Metric hyperparameters
        experiment.params.set(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            model="simple_nn"
        )

        experiment.log("Starting training", level="info")

        # Training loop
        for epoch in range(10):
            # Simulate training
            train_loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
            val_loss = 1.2 / (epoch + 1) + random.uniform(-0.05, 0.05)
            accuracy = min(0.95, 0.5 + epoch * 0.05)

            # Log metrics
            experiment.metrics.log(
                epoch=epoch,
                train=dict(loss=train_loss, accuracy=accuracy),
                eval=dict(loss=val_loss, accuracy=accuracy)
            )

            # Log progress
            experiment.log(
                f"Epoch {epoch + 1}/10 complete",
                level="info",
                metadata={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy
                }
            )

        experiment.log("Training complete!", level="info")
        print(f"✓ Experiment saved to: {experiment._storage.root_path}")

if __name__ == "__main__":
    train_simple_model()
```

## Example 2: PyTorch MNIST Training

Complete PyTorch MNIST training with full experiment tracking.

```python
"""PyTorch MNIST training with ML-Dash tracking."""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from ml_dash import Experiment

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def train_mnist():
    """Train MNIST model with ML-Dash tracking."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ML-Dash experiment
    with Experiment(
        name="mnist-pytorch",
        project="computer-vision",
        description="MNIST classification with PyTorch",
        tags=["mnist", "pytorch", "classification"],
        local_path=".dash"
    ).run as experiment:
        # Metric configuration
        experiment.params.set({
            "model": {
                "architecture": "SimpleMLP",
                "layers": [784, 128, 64, 10]
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            },
            "device": str(device),
            "dataset": "MNIST"
        })

        experiment.log(f"Training on {device}", level="info")

        best_accuracy = 0.0

        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            avg_val_loss = val_loss / len(test_loader)
            val_accuracy = correct / total

            # Log metrics
            experiment.metrics.log(
                epoch=epoch,
                train=dict(loss=avg_train_loss, accuracy=train_accuracy),
                eval=dict(loss=avg_val_loss, accuracy=val_accuracy)
            )

            # Log progress
            experiment.log(
                f"Epoch {epoch + 1}/{epochs}",
                level="info",
                metadata={
                    "train_loss": avg_train_loss,
                    "train_acc": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_acc": val_accuracy
                }
            )

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), "best_model.pth")
                experiment.files("models").save(
                    "best_model.pth",
                    description=f"Best model (accuracy: {best_accuracy:.4f})",
                    tags=["best"],
                    metadata={"epoch": epoch, "accuracy": best_accuracy}
                )
                experiment.log(f"New best model saved: {best_accuracy:.4f}", level="info")

        # Save final model
        torch.save(model.state_dict(), "final_model.pth")
        experiment.files("models").save(
            "final_model.pth",
            description="Final model after all epochs",
            tags=["final"]
        )

        experiment.log("Training complete!", level="info")
        print(f"✓ Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_mnist()
```

## Example 3: Hyperparameter Search

Grid search over hyperparameters with full tracking.

```python
"""Hyperparameter search example."""
from itertools import product
import random
from ml_dash import Experiment

def train_with_config(lr, batch_size, experiment):
    """Simulate training with given hyperparameters."""
    # Simulate training
    epochs = 10
    for epoch in range(epochs):
        # Lower learning rate = better but slower convergence
        loss = 1.0 / (epoch + 1) * (lr / 0.01) + random.uniform(-0.05, 0.05)
        # Larger batch size = slightly worse performance
        accuracy = min(0.95, 0.5 + epoch * 0.05 * (32 / batch_size))

        experiment.metrics.log(
            epoch=epoch,
            train=dict(loss=loss, accuracy=accuracy)
        )

    return accuracy

def hyperparameter_search():
    """Run grid search over hyperparameters."""

    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 32, 64]

    results = []

    for lr, bs in product(learning_rates, batch_sizes):
        experiment_name = f"search-lr{lr}-bs{bs}"

        with Experiment(
            name=experiment_name,
            project="hyperparameter-search",
            description=f"Grid search: lr={lr}, batch_size={bs}",
            tags=["grid-search", f"lr-{lr}", f"bs-{bs}"],
            local_path=".dash"
        ).run as experiment:
            # Metric hyperparameters
            experiment.params.set(
                learning_rate=lr,
                batch_size=bs,
                optimizer="sgd",
                epochs=10
            )

            experiment.log(f"Starting training with lr={lr}, bs={bs}")

            # Train
            final_accuracy = train_with_config(lr, bs, experiment)

            experiment.log(f"Final accuracy: {final_accuracy:.4f}")

            results.append({
                "lr": lr,
                "batch_size": bs,
                "accuracy": final_accuracy
            })

    # Find best configuration
    best = max(results, key=lambda x: x["accuracy"])
    print("\n" + "=" * 50)
    print("Hyperparameter Search Complete!")
    print("=" * 50)
    print(f"Best configuration:")
    print(f"  Learning rate: {best['lr']}")
    print(f"  Batch size: {best['batch_size']}")
    print(f"  Accuracy: {best['accuracy']:.4f}")

if __name__ == "__main__":
    hyperparameter_search()
```

## Example 4: Experiment Comparison

Metric multiple experiments and compare results.

```python
"""Compare multiple experiments."""
from ml_dash import Experiment
import random

def train_model(architecture, experiment):
    """Train model with given architecture."""
    epochs = 20

    # Different architectures converge differently
    base_lr = {"cnn": 0.85, "resnet": 0.90, "vit": 0.92}[architecture]

    for epoch in range(epochs):
        acc = min(base_lr, 0.5 + epoch * (base_lr - 0.5) / epochs + random.uniform(-0.02, 0.02))
        experiment.metrics("train").log(accuracy=acc, epoch=epoch)

    return acc

def compare_architectures():
    """Compare different model architectures."""

    architectures = ["cnn", "resnet", "vit"]
    results = {}

    for arch in architectures:
        with Experiment(
            name=f"comparison-{arch}",
            project="architecture-comparison",
            description=f"Training {arch} on CIFAR-10",
            tags=["comparison", arch, "cifar10"],
            local_path=".dash"
        ).run as experiment:
            # Configuration
            experiment.params.set(
                architecture=arch,
                dataset="cifar10",
                batch_size=128,
                epochs=20
            )

            experiment.log(f"Training {arch} architecture")

            # Train
            final_acc = train_model(arch, experiment)

            experiment.log(f"Final accuracy: {final_acc:.4f}")

            results[arch] = final_acc

    # Print comparison
    print("\n" + "=" * 50)
    print("Architecture Comparison Results")
    print("=" * 50)
    for arch in sorted(results.keys(), key=lambda x: results[x], reverse=True):
        print(f"{arch:10s}: {results[arch]:.4f}")

if __name__ == "__main__":
    compare_architectures()
```

## Example 5: Logging and Debugging

Comprehensive logging for debugging.

```python
"""Debugging example with comprehensive logging."""
from ml_dash import Experiment
import random

def train_with_debug():
    """Training with extensive debugging logs."""

    with Experiment(
        name="debug-training",
        project="debugging",
        description="Training with debug logging",
        tags=["debug"],
        local_path=".dash"
    ).run as experiment:
        experiment.params.set(
            learning_rate=0.001,
            batch_size=32,
            model="debug_net"
        )

        experiment.log("Training experiment started", level="info")
        experiment.log("Initializing model", level="debug")

        # Simulate some issues
        for epoch in range(5):
            experiment.log(f"Starting epoch {epoch + 1}", level="debug")

            # Simulate varying behavior
            loss = 1.0 / (epoch + 1)

            if epoch == 2:
                # Simulate a warning
                experiment.log(
                    "Learning rate may be too high",
                    level="warn",
                    metadata={"current_lr": 0.001, "suggested_lr": 0.0001}
                )

            if random.random() < 0.2:
                # Simulate occasional error
                experiment.log(
                    "Gradient clipping applied",
                    level="warn",
                    metadata={"gradient_norm": 15.5, "max_norm": 10.0}
                )

            experiment.metrics("train").log(loss=loss, epoch=epoch)

            experiment.log(
                f"Epoch {epoch + 1} complete",
                level="info",
                metadata={"loss": loss}
            )

        experiment.log("Training complete", level="info")

if __name__ == "__main__":
    train_with_debug()
```

## Running the Examples

All examples are available in the `docs/examples/` directory:

```bash
# Navigate to docs/examples
cd docs/examples

# Run any example
python complete_training.py
python hyperparameter_search.py
python architecture_comparison.py
```

## See Also

**Feature-specific guides:**
- [Logging](logging.md) - Structured logging with levels and metadata
- [Parameters](parameters.md) - Hyperparameter tracking and flattening
- [Metrics](metrics.md) - Time-series metrics tracking
- [Files](files.md) - File upload and management

**Deployment & Operations:**
- **[Deployment Guide](deployment.md)** - Deploy your own ML-Dash server
- **[Architecture](architecture.md)** - How ML-Dash works internally
- **[FAQ & Troubleshooting](faq.md)** - Common problems and solutions

**Getting Started:**
- [Getting Started](getting-started.md) - Quick start tutorial
- [API Quick Reference](api-quick-reference.md) - API cheat sheet
