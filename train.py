"""
train.py
--------
Training script for EEG anomaly classification using CWT scalogram images.

Supports two modes:
- three_class : Normal, Slowing Waves, Spike and Sharp Waves (3 classes)
- binary      : Normal vs Abnormal (2 classes)

Usage:
    python train.py
    python train.py --config configs/training.yaml
    python train.py --mode binary --model googlenet --epochs 20
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import (
    GoogLeNet,
    VGG16,
    EfficientNetB1,
)


LABEL_CLASSES_3 = ["Normal", "Slowing Waves", "Spike and Sharp waves"]
LABEL_CLASSES_2 = ["Normal", "Abnormal"]


class EEGCWTDataset(Dataset):
    """PyTorch Dataset for EEG CWT scalogram images.

    Loads images from directory structure:
        root/
        ├── Normal/
        │   ├── img_0_0_file.png
        │   └── ...
        ├── Slowing Waves/
        └── Spike and Sharp waves/

    For binary mode, Slowing Waves and Spike and Sharp waves are merged
    into a single "Abnormal" class (label 1).

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing class subdirectories.
    mode     : str
        "three_class" or "binary".
    transform: callable, optional
        torchvision transforms to apply to images.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "three_class",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform

        if mode == "binary":
            self.class_names = LABEL_CLASSES_2
            self.label_map = {
                "Normal": 0,
                "Slowing Waves": 1,
                "Spike and Sharp waves": 1,
            }
        else:
            self.class_names = LABEL_CLASSES_3
            self.label_map = {
                "Normal": 0,
                "Slowing Waves": 1,
                "Spike and Sharp waves": 2,
            }

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Build list of (image_path, label) tuples."""
        samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            label = self.label_map[class_name]
            for img_path in class_dir.glob("*.png"):
                samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_config(config_path: str | Path) -> Dict:
    """Load training configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    training = config.get("training", {})

    if "num_classes" not in training:
        mode = training.get("mode", "three_class")
        training["num_classes"] = 3 if mode == "three_class" else 2

    return config


def create_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function to create a model by name.

    Args:
        model_name   : "vgg16", "googlenet", or "efficientnetb1".
        num_classes  : Number of output classes (default: 3).
        pretrained   : Whether to load pretrained weights (default: True).

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.lower()
    if model_name == "vgg16":
        return VGG16(num_classes=num_classes)
    elif model_name == "googlenet":
        return GoogLeNet(num_classes=num_classes)
    elif model_name in ("efficientnetb1", "efficientnet_b1"):
        return EfficientNetB1(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose from: vgg16, googlenet, efficientnetb1"
        )


def get_default_transforms() -> Dict[str, transforms.Compose]:
    """Return default image transforms for training and validation."""
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }


def compute_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        dataloader : Training dataloader.
        num_classes: Number of classes.

    Returns:
        Tensor of class weights (inverse frequency).
    """
    counts = [0] * num_classes
    for _, labels in dataloader:
        for label in labels:
            counts[label.item()] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            loss_main = criterion(outputs[0], labels)
            loss_aux = criterion(outputs[1], labels) if len(outputs) > 1 else 0
            loss = loss_main + 0.3 * loss_aux
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = (
            outputs[0] if isinstance(outputs, tuple) else torch.max(outputs, 1)
        )
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    checkpoint_dir: Path,
    model_name: str,
    mode: str,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint.

    Args:
        model         : PyTorch model.
        epoch         : Current epoch number.
        checkpoint_dir: Directory to save checkpoints.
        model_name    : Name of the model architecture.
        mode          : Training mode ("three_class" or "binary").
        is_best       : Whether this is the best model so far.

    Returns:
        Path to the saved checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{mode}_e{epoch}.pt"
    filepath = checkpoint_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_name": model_name,
            "mode": mode,
        },
        filepath,
    )

    if is_best:
        best_path = checkpoint_dir / f"{model_name}_{mode}_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": model_name,
                "mode": mode,
            },
            best_path,
        )

    return filepath


def main(config_path: str = "configs/training.yaml") -> None:
    """Main training function."""
    config = load_config(config_path)
    training = config["training"]

    mode = training.get("mode", "three_class")
    model_name = training.get("model", "vgg16")
    num_classes = training.get("num_classes", 3 if mode == "three_class" else 2)
    learning_rate = training.get("learning_rate", 0.0001)
    epochs = training.get("epochs", 30)
    batch_size = training.get("batch_size", 32)
    checkpoint_dir = training.get("checkpoint_dir", "checkpoints")

    early_stopping = training.get("early_stopping", {})
    patience = early_stopping.get("patience", 5)
    min_delta = early_stopping.get("min_delta", 0.001)

    data_config = training.get("data", {})
    train_dir = data_config.get("train_dir", "data/train")
    valid_dir = data_config.get("valid_dir", "data/valid")
    test_dir = data_config.get("test_dir", "data/test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {mode}, Classes: {num_classes}, Model: {model_name}")

    transforms_dict = get_default_transforms()

    train_dataset = EEGCWTDataset(
        train_dir, mode=mode, transform=transforms_dict["train"]
    )
    valid_dataset = EEGCWTDataset(
        valid_dir, mode=mode, transform=transforms_dict["valid"]
    )

    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = create_model(model_name, num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        epoch_time = time.time() - start_time

        is_best = val_loss < best_val_loss - min_delta
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint_path = save_checkpoint(
            model,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            checkpoint_dir,
            model_name,
            mode,
            is_best=is_best,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s | "
            f"{'BEST' if is_best else ''}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    print(f"\nTraining complete. Checkpoints saved to: {checkpoint_dir}")
    print(f"Best model: {checkpoint_dir}/{model_name}_{mode}_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EEG anomaly classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["three_class", "binary"],
        help="Override training mode from config",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vgg16", "googlenet", "efficientnetb1"],
        help="Override model from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config",
    )

    args = parser.parse_args()

    if args.mode or args.model or args.epochs:
        import copy

        config = load_config(args.config)
        if args.mode:
            config["training"]["mode"] = args.mode
            config["training"]["num_classes"] = 3 if args.mode == "three_class" else 2
        if args.model:
            config["training"]["model"] = args.model
        if args.epochs:
            config["training"]["epochs"] = args.epochs

        training = config["training"]
        mode = training.get("mode", "three_class")
        model_name = training.get("model", "vgg16")
        num_classes = training.get("num_classes", 3 if mode == "three_class" else 2)
        learning_rate = training.get("learning_rate", 0.0001)
        epochs = training.get("epochs", 30)
        batch_size = training.get("batch_size", 32)
        checkpoint_dir = training.get("checkpoint_dir", "checkpoints")
        early_stopping = training.get("early_stopping", {})
        patience = early_stopping.get("patience", 5)
        min_delta = early_stopping.get("min_delta", 0.001)
        data_config = training.get("data", {})
        train_dir = data_config.get("train_dir", "data/train")
        valid_dir = data_config.get("valid_dir", "data/valid")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Mode: {mode}, Classes: {num_classes}, Model: {model_name}")

        transforms_dict = get_default_transforms()
        train_dataset = EEGCWTDataset(
            train_dir, mode=mode, transform=transforms_dict["train"]
        )
        valid_dataset = EEGCWTDataset(
            valid_dir, mode=mode, transform=transforms_dict["valid"]
        )

        print(
            f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}"
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = create_model(model_name, num_classes=num_classes).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        class_weights = compute_class_weights(train_loader, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, valid_loader, criterion, device)
            epoch_time = time.time() - start_time

            is_best = val_loss < best_val_loss - min_delta
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            checkpoint_path = save_checkpoint(
                model,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                checkpoint_dir,
                model_name,
                mode,
                is_best=is_best,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s | "
                f"{'BEST' if is_best else ''}"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        print(f"\nTraining complete. Checkpoints saved to: {checkpoint_dir}")
        print(f"Best model: {checkpoint_dir}/{model_name}_{mode}_best.pt")
    else:
        main(args.config)
