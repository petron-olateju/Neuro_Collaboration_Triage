"""
run_transparency.py
-------------------
Standalone script to generate transparency heatmaps for uncertain EEG samples.

This script:
1. Loads a trained model and test data
2. Identifies uncertain samples (AI confidence below threshold)
3. Generates heatmaps showing:
   - Class difference attribution (regions associated with abnormal patterns)
   - Uncertainty attribution (regions where model is confused)
4. Creates comparison visualizations

Usage:
    python run_transparency.py --checkpoint checkpoints/nmt_resnet18_three_class_best.pt
    
    # With custom settings
    python run_transparency.py --checkpoint checkpoints/nmt_resnet18_three_class_best.pt \
        --confidence 0.6 --num-samples 20 --methods both
"""

import argparse
import os
import glob

import torch
from torch.utils.data import DataLoader
import yaml

from train import EEGCWTDataset, create_model, get_default_transforms
from utils.dataset_with_metadata import EEGCWTMetadataDataset
from utils.transparency_module import (
    generate_transparency_report,
)

# Import from test scripts
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_and_collab_sweep import get_test_subjects_from_config

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_DATASET = "nmt"
DEFAULT_CONFIG = "configs/dataset_config.yaml"


def find_checkpoints(checkpoint_dir, dataset):
    """Find all model checkpoints for a dataset."""
    pattern = os.path.join(checkpoint_dir, f"{dataset}_*_best.pt")
    files = glob.glob(pattern)
    return sorted(files)


def parse_checkpoint_filename(filename, dataset):
    """Parse checkpoint filename to extract model and mode."""
    basename = os.path.basename(filename)
    name_without_ext = basename.replace("_best.pt", "")
    parts = name_without_ext.split("_")

    if len(parts) >= 3 and parts[0] == dataset:
        model_name = parts[1]
        mode = parts[2]

        if "three" in mode:
            mode = "three_class"
            num_classes = 3
        elif "binary" in mode:
            mode = "binary"
            num_classes = 2
        else:
            return None, None, None

        return model_name, mode, num_classes

    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate transparency heatmaps for uncertain EEG samples"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint (or use --checkpoint-dir for auto-discovery)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Uncertainty threshold - samples below this confidence are considered uncertain",
    )
    parser.add_argument(
        "--max-samples-scan",
        type=int,
        default=5000,
        help="Maximum number of samples to scan to find uncertain ones (default: 5000)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of uncertain samples to visualize",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["class_diff", "uncertainty", "both"],
        help="Attribution methods: class_diff, uncertainty, or both",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/transparency/",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="three_class",
        choices=["three_class", "binary"],
        help="Model mode",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="three_class",
        choices=["binary", "three_class", "all"],
        help="Which modes to process (when using auto-discovery)",
    )
    parser.add_argument(
        "--config-subjects",
        action="store_true",
        default=True,
        help="Use abnormal subjects from config for testing",
    )

    args = parser.parse_args()

    confidence_threshold = args.confidence
    max_samples_scan = args.max_samples_scan
    num_samples = args.num_samples
    methods = args.methods
    if methods == "both":
        methods = ["class_diff", "uncertainty"]
    output_dir = args.output

    print(f"Dataset: {args.dataset}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Number of samples: {num_samples}")
    print(f"Methods: {methods}")
    print(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transforms_dict = get_default_transforms()

    # Determine checkpoints to process
    if args.checkpoint:
        checkpoint_files = [args.checkpoint]
    else:
        checkpoint_files = find_checkpoints(checkpoint_dir, dataset)
        # Filter by mode
        if args.modes == "three_class":
            checkpoint_files = [
                f for f in checkpoint_files if "three" in os.path.basename(f)
            ]
        elif args.modes == "binary":
            checkpoint_files = [
                f for f in checkpoint_files if "binary" in os.path.basename(f)
            ]

    if not checkpoint_files:
        print(f"No checkpoint files found")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    for checkpoint_path in checkpoint_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(checkpoint_path)}")

        # Parse checkpoint
        model_name, mode, num_classes = parse_checkpoint_filename(
            checkpoint_path, args.dataset
        )

        if model_name is None:
            print(f"Skipping - could not parse model/mode")
            continue

        checkpoint_key = os.path.basename(checkpoint_path).replace("_best.pt", "")
        print(f"Model: {model_name}, Mode: {mode}, Num classes: {num_classes}")

        # Load model
        try:
            model = create_model(model_name, num_classes=num_classes).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        # Load test dataset
        try:
            if args.config_subjects:
                test_subject_ids = get_test_subjects_from_config(
                    DEFAULT_CONFIG, args.dataset
                )
                print(f"Using {len(test_subject_ids)} config-based test subjects")
            else:
                test_subject_ids = None

            test_dir = f"data/{args.dataset}/test"
            base_dataset = EEGCWTDataset(
                test_dir,
                mode=mode,
                transform=transforms_dict["test"],
                subject_ids=test_subject_ids,
            )
            test_dataset = EEGCWTMetadataDataset(base_dataset, "data/nmt_metadata.csv")
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Create checkpoint-specific output directory
        checkpoint_output_dir = os.path.join(output_dir, checkpoint_key)

        # Generate transparency report
        metadata = generate_transparency_report(
            model=model,
            test_loader=test_loader,
            confidence_threshold=confidence_threshold,
            num_samples=num_samples,
            max_samples_scan=max_samples_scan,
            methods=methods,
            output_dir=checkpoint_output_dir,
            device=device,
        )

    print("\n" + "=" * 60)
    print("Transparency generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
