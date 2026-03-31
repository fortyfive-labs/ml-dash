---
sidebar_label: PyTorch MNIST
---

# PyTorch MNIST Training

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=batch_size
    )

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    with Experiment(
        prefix="alice/computer-vision/mnist-pytorch",
        description="MNIST classification with PyTorch",
        tags=["mnist", "pytorch", "classification"]
    ).run as experiment:
        experiment.params.set({
            "model": {"architecture": "SimpleMLP", "layers": [784, 128, 64, 10]},
            "training": {"optimizer": "adam", "learning_rate": learning_rate,
                         "batch_size": batch_size, "epochs": epochs},
            "device": str(device),
            "dataset": "MNIST"
        })

        best_accuracy = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    correct += output.argmax(dim=1).eq(target).sum().item()
                    total += target.size(0)

            avg_val_loss = val_loss / len(test_loader)
            val_accuracy = correct / total

            experiment.metrics.log(
                epoch=epoch,
                train=dict(loss=avg_train_loss, accuracy=train_accuracy),
                eval=dict(loss=avg_val_loss, accuracy=val_accuracy)
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), "best_model.pth")
                experiment.files("models").save(
                    "best_model.pth",
                    description=f"Best model (accuracy: {best_accuracy:.4f})",
                    tags=["best"],
                    metadata={"epoch": epoch, "accuracy": best_accuracy}
                )

        torch.save(model.state_dict(), "final_model.pth")
        experiment.files("models").save("final_model.pth", tags=["final"])
        print(f"Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_mnist()
```
