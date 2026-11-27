
import torch
import torch.nn as nn
from torchvision import models


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetClassifier, self).__init__()

        # Load MobileNetV2 pretrained on ImageNet
        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = models.mobilenet_v2(weights=weights)

        # Freeze pretrained feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace the original classifier
        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileNetClassifier(num_classes=2).to(device)

    sample_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(sample_input)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Device:", device)
    print("Output shape:", output.shape)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("MobileNetV2 test successful!")
