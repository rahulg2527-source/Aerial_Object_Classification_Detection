
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.custom_cnn import CustomCNN
from src.data.preprocessing import train_loader, valid_loader, DEVICE


# Training configuration
EPOCHS = 15
LEARNING_RATE = 0.001
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_custom_cnn.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():
    print("Device:", DEVICE)
    print("Starting Custom CNN training...")

    model = CustomCNN(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_valid_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        valid_loss, valid_accuracy = evaluate(
            model,
            valid_loader,
            criterion,
            DEVICE
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": ["bird", "drone"],
                    "best_valid_accuracy": best_valid_accuracy,
                    "epoch": epoch + 1,
                },
                BEST_MODEL_PATH
            )

            print("Best model saved!")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_valid_accuracy * 100:.2f}%")
    print(f"Saved model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
