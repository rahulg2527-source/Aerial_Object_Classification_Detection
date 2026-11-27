from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolo"
    / "bird_drone_yolov8n"
    / "weights"
    / "best.pt"
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"


class ObjectDetector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

    def predict(self, image: Image.Image):
        image = image.convert("RGB")

        results = self.model.predict(
            source=image,
            imgsz=416,
            conf=0.50,
            iou=0.7,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        annotated_array = result.plot()
        annotated_image = Image.fromarray(
            annotated_array[:, :, ::-1]
        )

        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "box": [round(value, 2) for value in coordinates]
            })

        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "count": len(detections)
        }