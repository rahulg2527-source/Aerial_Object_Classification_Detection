
from pathlib import Path
import sys

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.custom_cnn import CustomCNN
from src.data.preprocessing import test_loader, DEVICE


MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_custom_cnn.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model():
    print("Evaluation device:", DEVICE)

    # Load trained checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    # Initialize model
    model = CustomCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    # Evaluate on test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n========== TEST RESULTS ==========")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Save classification report
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)

    # Generate confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Custom CNN - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    confusion_path = PLOT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close()

    print("Classification report saved:", report_path)
    print("Confusion matrix saved:", confusion_path)


if __name__ == "__main__":
    evaluate_model()
