
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.mobilenet_model import MobileNetClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "best_mobilenet_v2.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ImageClassifier:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classification model not found: {MODEL_PATH}"
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        self.class_names = checkpoint["class_names"]

        self.model = MobileNetClassifier(
            num_classes=len(self.class_names),
            pretrained=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        return {
            "class_name": self.class_names[predicted_index.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                name: float(probabilities[0][index].item())
                for index, name in enumerate(self.class_names)
            }
        }
