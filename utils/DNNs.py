"""
DNNs.py
-------
PyTorch nn.Module implementations of deep neural networks for EEG anomaly
classification, translated from the TensorFlow/Keras source in
artifacts/source_code_files/deeplearning.ipynb.

Models
------
- VGG16        : Standard VGG-16 architecture (13 conv + 3 FC layers).
- GoogLeNet    : Custom Inception network with auxiliary classifiers.
- EfficientNetB1 : Pre-trained EfficientNet-B1 (timm) with custom classifier.

Input shape : (B, 3, 224, 224)
Output      : 3 classes (Normal, Slowing Waves, Spike and Sharp Waves)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Inception module used inside GoogLeNet.

    Args:
        in_channels  : Number of input channels.
        f1           : Number of 1×1 convolution filters.
        f2_conv1    : 1×1 conv filters before the 3×3 branch.
        f2_conv3    : 3×3 conv filters on the 3×3 branch.
        f3_conv1    : 1×1 conv filters before the 5×5 branch.
        f3_conv5    : 5×5 conv filters on the 5×5 branch.
        f4          : 1×1 conv filters after the max-pool branch.
    """

    def __init__(
        self,
        in_channels: int,
        f1: int,
        f2_conv1: int,
        f2_conv3: int,
        f3_conv1: int,
        f3_conv5: int,
        f4: int,
    ) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, f2_conv1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2_conv1, f2_conv3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, f3_conv1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3_conv1, f3_conv5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, f4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception-v1) for EEG classification.

    Translated from the TensorFlow/Keras implementation in deeplearning.ipynb.
    Produces three outputs: main output and two auxiliary classifier outputs.
    Use with a loss that sums all three outputs.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.inception3a = InceptionBlock(
            192, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32
        )
        self.inception3b = InceptionBlock(
            256, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64
        )
        self.inception4a = InceptionBlock(
            480, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64
        )
        self.inception4b = InceptionBlock(
            512, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64
        )
        self.inception4c = InceptionBlock(
            512, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64
        )
        self.inception4d = InceptionBlock(
            512, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64
        )
        self.inception4e = InceptionBlock(
            528, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128
        )
        self.inception5a = InceptionBlock(
            832, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128
        )
        self.inception5b = InceptionBlock(
            832, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128
        )

        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc_main = nn.Linear(1024, num_classes)

        self.aux1 = self._make_aux_classifier(in_channels=512)
        self.aux2 = self._make_aux_classifier(in_channels=528)

    def _make_aux_classifier(self, in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool5(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool5(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main = self.fc_main(x)

        if self.training:
            return main, aux1
        return main


class VGG16(nn.Module):
    """VGG-16 for EEG classification.

    Translated from keras.applications.vgg16.VGG16.
    Architecture: 5 conv blocks (2+2+3+3+3 conv layers) + 3 FC layers.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return x


class EfficientNetB1(nn.Module):
    """EfficientNet-B1 with a custom classifier head for EEG classification.

    Uses timm's pre-trained EfficientNet-B1 (ImageNet weights).
    The original notebook used keras.applications.efficientnet_v2.EfficientNetV1B1.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for EfficientNetB1. Install with: pip install timm"
            )

        self.backbone = timm.create_model(
            "efficientnet_b1",
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
