import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.classification import ImageClassifier


classifier = ImageClassifier()

test_dir = PROJECT_ROOT / "data" / "raw" / "classification" / "test"

for class_name in ["bird", "drone"]:
    class_dir = test_dir / class_name
    image_path = next(class_dir.glob("*"))

    image = Image.open(image_path)
    result = classifier.predict(image)

    print(f"\nActual class: {class_name}")
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")