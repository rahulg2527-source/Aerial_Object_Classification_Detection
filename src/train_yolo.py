
from pathlib import Path
import torch
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "detection"
    / "dataset"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"

# Training configuration
MODEL_NAME = "yolov8n.pt"
IMAGE_SIZE = 416
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print("========== YOLOv8 TRAINING ==========")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Dataset:", DATA_YAML)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    # Load pretrained YOLOv8 nano model
    model = YOLO(MODEL_NAME)

    # Start training
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        project=str(OUTPUT_DIR),
        name="bird_drone_yolov8n",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        save=True,
        plots=True,
        amp=True,
        cache=False,
        seed=42,
        deterministic=True
    )

    print("\n========== TRAINING COMPLETED ==========")
    print("Best model:", OUTPUT_DIR / "bird_drone_yolov8n" / "weights" / "best.pt")
    print("Last model:", OUTPUT_DIR / "bird_drone_yolov8n" / "weights" / "last.pt")
    print("Training results:", results)


if __name__ == "__main__":
    main()
