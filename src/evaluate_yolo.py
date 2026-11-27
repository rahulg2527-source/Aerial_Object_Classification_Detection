
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

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "detection"
    / "dataset"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"

def main():
    print("========== YOLOv8 TEST EVALUATION ==========")
    print("CUDA available:", torch.cuda.is_available())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="bird_drone_test_evaluation",
        exist_ok=True
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\nEvaluation completed successfully.")
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
