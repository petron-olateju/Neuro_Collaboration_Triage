import os
import glob
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from sklearn.metrics import classification_report

# Import from train.py
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import EEGCWTDataset, create_model, get_default_transforms, load_config

# Import from utils
from utils.dataset_with_metadata import EEGCWTMetadataDataset, classify_age_group


# --- CONFIGURATION ---
DEFAULT_DATASET = "nmt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_METADATA_CSV = "data/nmt_metadata.csv"


# --- COLLABORATION LOGIC ---


def get_metadata_item(metadata_list, i):
    """Extract metadata for sample i, handling both list-of-dicts and dict-of-lists formats."""
    if isinstance(metadata_list, dict):
        # DataLoader collated as dict of lists
        age_val = metadata_list["age"][i]
        if hasattr(age_val, "item"):
            age_val = age_val.item()
        return {
            "subject_id": metadata_list["subject_id"][i],
            "gender": metadata_list["gender"][i],
            "age": age_val,
        }
    elif isinstance(metadata_list, list) and len(metadata_list) > i:
        item = metadata_list[i]
        if isinstance(item, dict):
            return item
        elif isinstance(item, str):
            parts = item.split("_")
            subject_id = parts[0] if parts else "unknown"
            return {"subject_id": subject_id, "gender": "unknown", "age": -1}
        else:
            return {"subject_id": str(item), "gender": "unknown", "age": -1}
    else:
        return {"subject_id": "unknown", "gender": "unknown", "age": -1}


def apply_strategy_a(y_true, y_probs, confidence_threshold, metadata_list):
    """
    Apply Strategy A: Confidence-based deferral.
    Returns predictions and decisions.
    """
    confidences, predictions = torch.max(y_probs, dim=1)

    results = []

    for i in range(len(predictions)):
        decision = "AI" if confidences[i] >= confidence_threshold else "HUMAN"
        final_label = predictions[i].item() if decision == "AI" else y_true[i].item()

        meta = get_metadata_item(metadata_list, i)

        results.append(
            {
                "y_true": y_true[i].item(),
                "y_pred": final_label,
                "confidence": confidences[i].item(),
                "decision": decision,
                "subject_id": meta.get("subject_id", "unknown"),
                "gender": meta.get("gender", "unknown"),
                "age": meta.get("age", -1),
            }
        )

    return results


def apply_strategy_b(y_true, y_probs, cost_alpha, metadata_list):
    """
    Apply Strategy B: Cost-aware deferral (risk-based).
    Returns predictions and decisions.
    """
    results = []

    for i in range(len(y_probs)):
        prob_pathology = 1 - y_probs[i][0]

        decision = "HUMAN" if prob_pathology > cost_alpha else "AI"

        # Get prediction from model
        _, predictions = torch.max(y_probs, dim=1)
        final_label = predictions[i].item() if decision == "AI" else y_true[i].item()

        meta = get_metadata_item(metadata_list, i)

        results.append(
            {
                "y_true": y_true[i].item(),
                "y_pred": final_label,
                "prob_pathology": prob_pathology.item(),
                "decision": decision,
                "subject_id": meta.get("subject_id", "unknown"),
                "gender": meta.get("gender", "unknown"),
                "age": meta.get("age", -1),
            }
        )

    return results


# --- FAIRNESS ANALYSIS ---


def compute_group_metrics(results_list, group_key):
    """
    Compute metrics for each group in the given key (e.g., 'gender' or 'age_group').
    """
    df = results_list.copy()

    # Add age group column
    df["age_group"] = df["age"].apply(classify_age_group)

    group_metrics = {}

    for group_name, group_df in df.groupby(group_key):
        if group_name == "unknown" or group_name == -1:
            continue

        y_true = group_df["y_true"].values
        y_pred = group_df["y_pred"].values
        decisions = group_df["decision"].values

        # Compute classification metrics
        report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        metrics = {
            "accuracy": round(report["accuracy"], 4),
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall": round(report["weighted avg"]["recall"], 4),
            "f1": round(report["weighted avg"]["f1-score"], 4),
        }

        # Compute escalation rate
        human_count = (decisions == "HUMAN").sum()
        escalation_rate = (
            (human_count / len(decisions)) * 100 if len(decisions) > 0 else 0
        )
        metrics["escalation_rate"] = round(escalation_rate, 2)

        # Sample count
        metrics["sample_count"] = int(len(group_df))

        group_metrics[str(group_name)] = metrics

    return group_metrics


def analyze_fairness(results_list):
    """
    Perform fairness analysis on results.
    Returns dict with results by gender and by age group.
    """
    results_df = results_list.copy()

    # Results by gender
    gender_metrics = compute_group_metrics(results_df, "gender")

    # Results by age group
    results_df["age_group"] = results_df["age"].apply(classify_age_group)
    age_metrics = compute_group_metrics(results_df, "age_group")

    # Overall metrics
    y_true = results_df["y_true"].values
    y_pred = results_df["y_pred"].values
    decisions = results_df["decision"].values

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    overall = {
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
    }
    human_count = (decisions == "HUMAN").sum()
    overall["escalation_rate"] = round((human_count / len(decisions)) * 100, 2)
    overall["sample_count"] = len(results_df)

    return {
        "overall": overall,
        "by_gender": gender_metrics,
        "by_age_group": age_metrics,
    }


# --- CHECKPOINT PARSING ---


def find_checkpoints(checkpoint_dir):
    """Find all model checkpoints."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    return checkpoint_files


def parse_checkpoint_filename(filename, dataset):
    """Parse checkpoint filename to extract model, mode, and num_classes info."""
    basename = os.path.basename(filename)
    name_without_ext = basename.replace("_best.pt", "")
    parts = name_without_ext.split("_")

    if len(parts) >= 3 and parts[0] == dataset:
        model = parts[1]
        mode = parts[2]
        num_classes = 3 if "three" in mode else 2
        return model, mode, num_classes

    return None, None, None


# --- INFERENCE ---


def run_inference_with_metadata(model, dataloader, device):
    """Run model inference and collect metadata."""
    model.eval()

    all_images = []
    all_labels = []
    all_probs = []
    all_metadata = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, metadata = batch
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)

            all_images.append(images.cpu())
            all_labels.append(labels)
            all_probs.append(probs)
            all_metadata.extend(metadata)

    y_true = torch.cat(all_labels)
    y_probs = torch.cat(all_probs)

    return y_true, y_probs, all_metadata


# --- MAIN ---


def main():
    parser = argparse.ArgumentParser(
        description="EEG Collaboration Fairness Analysis - Subgroup analysis by gender and age"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (default: nmt)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=DEFAULT_METADATA_CSV,
        help="Path to metadata CSV (default: data/nmt_metadata.csv)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Strategy A: defer to human if confidence below this threshold (default: 0.85)",
    )
    parser.add_argument(
        "--cost-alpha",
        type=float,
        default=0.15,
        help="Strategy B: defer to human if P(pathology) > this alpha (default: 0.15)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vgg16", "googlenet", "efficientnetb1", "vgg11", "resnet18"],
        help="Model name (use with --mode to specify checkpoint)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["binary", "three_class"],
        help="Mode (use with --model to specify checkpoint)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EEG Collaboration Fairness Analysis")
    print("=" * 60)

    dataset = args.dataset
    test_dir = f"data/{dataset}/test"
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    metadata_csv = args.metadata_csv

    confidence_threshold = args.confidence_threshold
    cost_alpha = args.cost_alpha

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Confidence threshold (Strategy A): {confidence_threshold}")
    print(f"Cost alpha (Strategy B): {cost_alpha}")

    # Check metadata CSV exists
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata CSV not found at {metadata_csv}")
        print("Run generate_metadata.py first to create the metadata file.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine which checkpoints to process
    if args.model and args.mode:
        # Specific checkpoint requested
        checkpoint_filename = f"{dataset}_{args.model}_{args.mode}_best.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            available = (
                os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
            )
            error_msg = f"Error: Checkpoint not found: {checkpoint_path}\n"
            if available:
                error_msg += f"Available checkpoints in {checkpoint_dir}:\n"
                for f in available:
                    error_msg += f"  - {f}\n"
            else:
                error_msg += f"No checkpoint files found in {checkpoint_dir}"
            print(error_msg)
            return

        checkpoint_files = [checkpoint_path]
        print(f"Processing specific checkpoint: {checkpoint_filename}")

    elif args.model or args.mode:
        # Partial args provided - error
        print("Error: Both --model and --mode must be provided together, or neither")
        return

    else:
        # Auto-discover all checkpoints
        checkpoint_files = find_checkpoints(checkpoint_dir)

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    # Load transforms
    transforms_dict = get_default_transforms()

    # Process each checkpoint
    for checkpoint_path in checkpoint_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {checkpoint_path}")
        print("=" * 60)

        # Parse checkpoint name
        model_name, mode, num_classes = parse_checkpoint_filename(
            checkpoint_path, dataset
        )

        if model_name is None:
            print(f"Skipping {checkpoint_path} - could not parse model/mode")
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

        # Load test dataset (base dataset)
        try:
            base_dataset = EEGCWTDataset(
                test_dir, mode=mode, transform=transforms_dict["test"]
            )
            # Wrap with metadata
            test_dataset = EEGCWTMetadataDataset(base_dataset, metadata_csv)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Run inference with metadata
        y_true, y_probs, metadata_list = run_inference_with_metadata(
            model, test_loader, device
        )

        print(f"Running fairness analysis...")

        # Strategy A: Confidence-based
        results_a = apply_strategy_a(
            y_true, y_probs, confidence_threshold, metadata_list
        )
        fairness_a = analyze_fairness(results_a)

        # Strategy B: Cost-aware
        results_b = apply_strategy_b(y_true, y_probs, cost_alpha, metadata_list)
        fairness_b = analyze_fairness(results_b)

        # Prepare output
        output_data = {
            "checkpoint": checkpoint_key,
            "model": model_name,
            "mode": mode,
            "num_classes": num_classes,
            "strategy_a": {
                "confidence_threshold": confidence_threshold,
                "results": fairness_a,
            },
            "strategy_b": {"cost_alpha": cost_alpha, "results": fairness_b},
        }

        # Save to YAML
        output_dir = "experiments"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"fairness_{checkpoint_key}.yaml")

        with open(output_file, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

        print(f"Fairness analysis saved to {output_file}")

        # Print summary
        print("\n--- Strategy A (Confidence-based) ---")
        print(
            f"Overall - Accuracy: {fairness_a['overall']['accuracy']}, Escalation: {fairness_a['overall']['escalation_rate']}%"
        )
        print(f"By Gender: {list(fairness_a['by_gender'].keys())}")
        print(f"By Age Group: {list(fairness_a['by_age_group'].keys())}")

        print("\n--- Strategy B (Cost-aware) ---")
        print(
            f"Overall - Accuracy: {fairness_b['overall']['accuracy']}, Escalation: {fairness_b['overall']['escalation_rate']}%"
        )
        print(f"By Gender: {list(fairness_b['by_gender'].keys())}")
        print(f"By Age Group: {list(fairness_b['by_age_group'].keys())}")

    print("\n" + "=" * 60)
    print("Fairness analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
