import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.detection import ObjectDetector


detector = ObjectDetector()

test_dir = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "detection"
    / "dataset"
    / "test"
    / "images"
)

image_path = next(test_dir.glob("*"))

image = Image.open(image_path)
result = detector.predict(image)

print(f"\nTest image: {image_path.name}")
print(f"Total detections: {result['count']}")

for detection in result["detections"]:
    print(
        f"Class: {detection['class_name']} | "
        f"Confidence: {detection['confidence']:.4f} | "
        f"Box: {detection['box']}"
    )

output_path = PROJECT_ROOT / "outputs" / "test_detection_result.jpg"
result["annotated_image"].save(output_path)

print(f"\nAnnotated image saved to: {output_path}")