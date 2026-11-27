from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolo"
    / "bird_drone_yolov8n"
    / "weights"
    / "best.pt"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "detection"
    / "dataset"
    / "test"
    / "images"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"


def main():
    print("========== YOLOv8 TEST PREDICTIONS ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Test images not found: {TEST_IMAGES}")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=416,
        conf=0.25,
        iou=0.7,
        device=0 if torch.cuda.is_available() else "cpu",
        batch=8,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name="bird_drone_test_predictions",
        exist_ok=True
    )

    print("\n========== PREDICTION SUMMARY ==========")
    print("Images processed:", len(results))
    print("Predictions saved to:")
    print(OUTPUT_DIR / "bird_drone_test_predictions")
    print("\nPrediction completed successfully.")


if __name__ == "__main__":
    main()