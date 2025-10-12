import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    AlexNet_Weights,
    VGG16_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    ConvNeXt_Tiny_Weights,
    MobileNet_V3_Small_Weights
)
def build_model(architecture='custom', input_shape=(3, 256, 256), pretrained=True):
    if architecture == 'custom':
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1]//4) * (input_shape[2]//4), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    elif architecture == 'alexnet':
        # Load with map_location to ensure weights are on the correct device
        weights = 'AlexNet_Weights.DEFAULT' if pretrained else None
        model = models.alexnet(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    elif architecture == 'vgg16':
        weights = 'VGG16_Weights.DEFAULT' if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    elif architecture == 'resnet50':
        weights = 'ResNet50_Weights.DEFAULT' if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    elif architecture == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    #  ConvNeXt-Tiny
    elif architecture == 'convnext_tiny':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        # Sostituisci l'ultimo livello di classificazione
        model.classifier[2] = nn.Sequential(
            nn.Linear(model.classifier[2].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    #  MobileNetV3-Small
    elif architecture == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    else:
        raise ValueError(f"Architettura non supportata: {architecture}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Explicitly move model to device (fixes device mismatch)
    model = model.to(device)
    return model