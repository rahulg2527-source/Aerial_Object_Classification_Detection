
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobilenet_model import MobileNetClassifier
from src.data.preprocessing import train_loader, valid_loader, DEVICE


# Training configuration
EPOCHS = 12
LEARNING_RATE = 0.001

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_mobilenet_v2.pth"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    print("Device:", DEVICE)
    print("Starting MobileNetV2 training...")

    model = MobileNetClassifier(
        num_classes=2,
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Only train parameters that are not frozen
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_valid_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        valid_loss, valid_accuracy = validate(
            model,
            valid_loader,
            criterion
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
                    "epoch": epoch + 1
                },
                BEST_MODEL_PATH
            )

            print("Best MobileNetV2 model saved!")

    print("\nMobileNetV2 training completed!")
    print(f"Best validation accuracy: {best_valid_accuracy * 100:.2f}%")
    print(f"Saved model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
