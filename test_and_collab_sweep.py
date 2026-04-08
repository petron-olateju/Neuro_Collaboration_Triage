import os
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

# Import from train.py
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import EEGCWTDataset, create_model, get_default_transforms, load_config
from utils.dataset_with_metadata import EEGCWTMetadataDataset, classify_age_group


# --- CONFIGURATION ---
CONFIDENCE_SWEEP_FILE = "experiments/confidence_threshold_sweep.yaml"
COST_SWEEP_FILE = "experiments/cost_alpha_sweep.yaml"
DEFAULT_DATASET = "nmt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_CONFIG = "configs/dataset_config.yaml"


def get_test_subjects_from_config(
    config_path: str = DEFAULT_CONFIG, dataset: str = "nmt"
):
    """Get test subjects from config file (abnormal subjects not in train/valid + test overlap)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_ids = set()
    for group in config["datasets"][dataset].get("train_subject_ids", []):
        train_ids.update(group)

    valid_ids = set()
    for group in config["datasets"][dataset].get("valid_subject_ids", []):
        valid_ids.update(group)

    test_ids = set()
    for group in config["datasets"][dataset].get("test_subject_ids", []):
        test_ids.update(group)

    # Get all abnormal EDF subjects
    edf_abnormal_dir = "eeg_data/Data/NMT_Events/edf/Abnormal EDF Files"
    if not os.path.exists(edf_abnormal_dir):
        return test_ids

    abnormal_subjects = set(
        int(f.replace(".edf", ""))
        for f in os.listdir(edf_abnormal_dir)
        if f.endswith(".edf")
    )

    # Filter out train + valid, keep test overlap
    test_subjects_to_exclude = train_ids | valid_ids
    available_for_test = abnormal_subjects - test_subjects_to_exclude
    overlapping = abnormal_subjects & test_ids

    return available_for_test | overlapping


# Default sweep ranges
DEFAULT_CONF_START = 0.0
DEFAULT_CONF_END = 1.0
DEFAULT_CONF_STEP = 0.05

DEFAULT_COST_START = 0.0
DEFAULT_COST_END = 1.0
DEFAULT_COST_STEP = 0.05


# --- COLLABORATION LOGIC (adapted for sweeps) ---


def apply_strategy_a(y_true, y_probs, confidence_threshold):
    """
    Apply Strategy A: Confidence-based deferral.
    Returns predictions and decisions.
    """
    confidences, predictions = torch.max(y_probs, dim=1)

    final_labels = []
    decisions = []

    for i in range(len(predictions)):
        if confidences[i] >= confidence_threshold:
            decisions.append("AI")
            final_labels.append(predictions[i].item())
        else:
            decisions.append("HUMAN")
            final_labels.append(y_true[i].item())

    return np.array(final_labels), decisions


def apply_strategy_b(y_true, y_probs, cost_alpha):
    """
    Apply Strategy B: Cost-aware deferral (risk-based).
    Returns predictions and decisions.
    """
    confidences, predictions = torch.max(y_probs, dim=1)
    final_labels = []
    decisions = []

    for i in range(len(y_probs)):
        prob_pathology = 1 - y_probs[i][0]

        if prob_pathology > cost_alpha:
            decisions.append("HUMAN")
            final_labels.append(y_true[i].item())
        else:
            decisions.append("AI")
            final_labels.append(predictions[i].item())

    return np.array(final_labels), decisions


# --- METRICS ---


def compute_metrics(y_true, y_pred, decisions=None):
    """Compute classification metrics."""
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
    }

    if decisions is not None:
        human_count = decisions.count("HUMAN")
        metrics["escalation_rate"] = round((human_count / len(decisions)) * 100, 2)

    return metrics


# --- FAIRNESS FUNCTIONS ---


def create_results_list(y_true_np, predictions, decisions, metadata_list):
    """Create list of dicts with metadata for fairness analysis."""
    results = []
    for i in range(len(predictions)):
        meta = metadata_list[i]
        gender = meta.get("gender", "unknown")
        if hasattr(gender, "item"):
            gender = gender.item()
        age = meta.get("age", -1)
        if hasattr(age, "item"):
            age = age.item()

        results.append(
            {
                "y_true": int(y_true_np[i]),
                "y_pred": int(predictions[i])
                if isinstance(predictions[i], (int, np.integer))
                else int(predictions[i].item()),
                "decision": decisions[i],
                "subject_id": str(meta.get("subject_id", "unknown")),
                "gender": str(gender),
                "age": int(age),
            }
        )
    return results


def compute_fairness_from_results(results_list):
    """Compute fairness metrics from results list."""
    if isinstance(results_list, list):
        df = pd.DataFrame(results_list)
    else:
        df = results_list.copy()

    # Compute by gender
    gender_metrics = {}
    for group_name, group_df in df.groupby("gender"):
        if str(group_name) == "unknown":
            continue
        y_t = group_df["y_true"].values
        y_p = group_df["y_pred"].values
        dec = group_df["decision"].values
        report = classification_report(y_t, y_p, zero_division=0, output_dict=True)
        human_count = (dec == "HUMAN").sum()
        gender_metrics[str(group_name)] = {
            "accuracy": round(float(report["accuracy"]), 4),
            "f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "escalation_rate": round(float(human_count / len(dec)) * 100, 2)
            if len(dec) > 0
            else 0,
            "sample_count": len(group_df),
        }

    # Compute by age_group
    df["age_group"] = df["age"].apply(classify_age_group)
    age_metrics = {}
    for group_name, group_df in df.groupby("age_group"):
        if str(group_name) == "unknown":
            continue
        y_t = group_df["y_true"].values
        y_p = group_df["y_pred"].values
        dec = group_df["decision"].values
        report = classification_report(y_t, y_p, zero_division=0, output_dict=True)
        human_count = (dec == "HUMAN").sum()
        age_metrics[str(group_name)] = {
            "accuracy": round(float(report["accuracy"]), 4),
            "f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "escalation_rate": round(float(human_count / len(dec)) * 100, 2)
            if len(dec) > 0
            else 0,
            "sample_count": len(group_df),
        }

    # Overall
    y_t = df["y_true"].values
    y_p = df["y_pred"].values
    dec = df["decision"].values
    report = classification_report(y_t, y_p, zero_division=0, output_dict=True)
    human_count = (dec == "HUMAN").sum()
    overall = {
        "accuracy": round(float(report["accuracy"]), 4),
        "f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "escalation_rate": round(float(human_count / len(dec)) * 100, 2)
        if len(dec) > 0
        else 0,
        "sample_count": len(df),
    }

    return {
        "overall": overall,
        "by_gender": gender_metrics,
        "by_age_group": age_metrics,
    }


# --- SWEEP FUNCTIONS ---


def sweep_confidence_thresholds(
    y_true,
    y_probs,
    thresholds,
    metadata_list=None,
    compute_fairness=False,
    cost_alpha_fixed=1.0,
):
    """
    Sweep confidence_threshold values while keeping cost_alpha fixed.
    With cost_alpha=1.0, Strategy B never triggers.
    """
    results = {}
    y_true_np = y_true.detach().cpu().numpy()
    predictions_np = torch.argmax(y_probs, dim=1).detach().cpu().numpy()

    for threshold in thresholds:
        labels, decisions = apply_strategy_a(y_true, y_probs, threshold)
        metrics = compute_metrics(y_true_np, labels, decisions)

        sweep_result = {"metrics": metrics}

        if compute_fairness and metadata_list is not None:
            results_list = create_results_list(
                y_true_np, labels, decisions, metadata_list
            )
            fairness = compute_fairness_from_results(results_list)
            sweep_result["fairness"] = fairness

        results[round(threshold, 2)] = sweep_result

    return results


def sweep_cost_alphas(
    y_true,
    y_probs,
    alphas,
    metadata_list=None,
    compute_fairness=False,
    confidence_threshold_fixed=0.0,
):
    """
    Sweep cost_alpha values while keeping confidence_threshold fixed.
    With confidence_threshold=0.0, Strategy A always accepts AI.
    """
    results = {}
    y_true_np = y_true.detach().cpu().numpy()

    for alpha in alphas:
        labels, decisions = apply_strategy_b(y_true, y_probs, alpha)
        metrics = compute_metrics(y_true_np, labels, decisions)

        sweep_result = {"metrics": metrics}

        if compute_fairness and metadata_list is not None:
            results_list = create_results_list(
                y_true_np, labels, decisions, metadata_list
            )
            fairness = compute_fairness_from_results(results_list)
            sweep_result["fairness"] = fairness

        results[round(alpha, 2)] = sweep_result

    return results


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
    """Run model inference on test data and return with metadata."""
    model.eval()
    all_labels = []
    all_probs = []
    all_metadata = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both (images, labels) and (images, labels, metadata) formats
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = [
                    {"subject_id": "unknown", "gender": "unknown", "age": -1}
                ] * len(labels)

            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)

            all_labels.append(labels)
            all_probs.append(probs)

            # Collect metadata per sample
            batch_size = labels.shape[0]
            for j in range(batch_size):
                # Handle both dict-of-lists (from DataLoader) and list-of-dicts
                if isinstance(metadata, dict):
                    gender = metadata["gender"][j]
                    if hasattr(gender, "item"):
                        gender = gender.item()
                    age = metadata["age"][j]
                    if hasattr(age, "item"):
                        age = age.item()
                    subject_id = metadata["subject_id"][j]
                    if hasattr(subject_id, "item"):
                        subject_id = subject_id.item()
                else:
                    meta = metadata[j]
                    gender = meta.get("gender", "unknown")
                    if hasattr(gender, "item"):
                        gender = gender.item()
                    age = meta.get("age", -1)
                    if hasattr(age, "item"):
                        age = age.item()
                    subject_id = meta.get("subject_id", "unknown")

                all_metadata.append(
                    {
                        "subject_id": str(subject_id),
                        "gender": str(gender),
                        "age": int(age),
                    }
                )

    y_true = torch.cat(all_labels)
    y_probs = torch.cat(all_probs)

    return y_true, y_probs, all_metadata


def run_inference(model, dataloader, device):
    """Run model inference on test data."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)

            all_labels.append(labels)
            all_probs.append(probs)

    y_true = torch.cat(all_labels)
    y_probs = torch.cat(all_probs)

    return y_true, y_probs


# --- MAIN ---


def main():
    parser = argparse.ArgumentParser(
        description="EEG Collaboration Sweep Analysis - Sweep collaboration parameters"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (default: nmt)",
    )
    parser.add_argument(
        "--confidence-start",
        type=float,
        default=DEFAULT_CONF_START,
        help="Confidence threshold sweep start (default: 0.0)",
    )
    parser.add_argument(
        "--confidence-end",
        type=float,
        default=DEFAULT_CONF_END,
        help="Confidence threshold sweep end (default: 1.0)",
    )
    parser.add_argument(
        "--confidence-step",
        type=float,
        default=DEFAULT_CONF_STEP,
        help="Confidence threshold sweep step (default: 0.05)",
    )
    parser.add_argument(
        "--cost-start",
        type=float,
        default=DEFAULT_COST_START,
        help="Cost alpha sweep start (default: 0.0)",
    )
    parser.add_argument(
        "--cost-end",
        type=float,
        default=DEFAULT_COST_END,
        help="Cost alpha sweep end (default: 1.0)",
    )
    parser.add_argument(
        "--cost-step",
        type=float,
        default=DEFAULT_COST_STEP,
        help="Cost alpha sweep step (default: 0.05)",
    )
    parser.add_argument(
        "--config-subjects",
        action="store_true",
        default=True,
        help="Use abnormal subjects from config for testing (default: True)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="three_class",
        choices=["binary", "three_class", "all"],
        help="Which modes to test: binary, three_class, or all (default: three_class)",
    )
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        default=False,
        help="Compute fairness metrics (by gender/age) at each sweep point",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EEG Collaboration Parameter Sweep")
    print("=" * 60)

    dataset = args.dataset
    test_dir = f"data/{dataset}/test"
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    use_config_subjects = args.config_subjects
    modes_to_test = args.modes
    include_fairness = args.include_fairness

    # Generate sweep values
    confidence_thresholds = np.arange(
        args.confidence_start,
        args.confidence_end + args.confidence_step,
        args.confidence_step,
    ).tolist()

    cost_alphas = np.arange(
        args.cost_start, args.cost_end + args.cost_step, args.cost_step
    ).tolist()

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Include fairness: {include_fairness}")
    print(
        f"Confidence thresholds: {len(confidence_thresholds)} values from {args.confidence_start} to {args.confidence_end}"
    )
    print(
        f"Cost alphas: {len(cost_alphas)} values from {args.cost_start} to {args.cost_end}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all checkpoints
    checkpoint_files = find_checkpoints(checkpoint_dir)

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    # Load transforms
    transforms_dict = get_default_transforms()

    # Initialize results dictionaries
    confidence_sweep_results = {}
    cost_sweep_results = {}

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

        # Skip modes based on --modes argument
        if modes_to_test == "three_class" and mode == "binary":
            print(
                f"Skipping {checkpoint_path} - binary mode skipped (use --modes all to run)"
            )
            continue
        if modes_to_test == "binary" and mode != "binary":
            print(
                f"Skipping {checkpoint_path} - three_class mode skipped (use --modes all to run)"
            )
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
            if use_config_subjects:
                test_subject_ids = get_test_subjects_from_config(
                    DEFAULT_CONFIG, dataset
                )
                print(f"Using {len(test_subject_ids)} config-based test subjects")
            else:
                test_subject_ids = None

            test_dataset = EEGCWTDataset(
                test_dir,
                mode=mode,
                transform=transforms_dict["test"],
                subject_ids=test_subject_ids,
            )

            # Wrap with metadata dataset if fairness is requested
            if include_fairness:
                test_dataset = EEGCWTMetadataDataset(
                    test_dataset, "data/nmt_metadata.csv"
                )

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Run inference with metadata if fairness requested
        if include_fairness:
            y_true, y_probs, metadata_list = run_inference_with_metadata(
                model, test_loader, device
            )
        else:
            y_true, y_probs = run_inference(model, test_loader, device)
            metadata_list = None

        print(f"Running sweeps...")

        # Compute baseline fairness if requested
        baseline_fairness = None
        if include_fairness and metadata_list is not None:
            # Baseline: AI always decides (no deferral)
            predictions = torch.argmax(y_probs, dim=1).detach().cpu().numpy()
            baseline_decisions = ["AI"] * len(predictions)
            y_true_np = y_true.detach().cpu().numpy()
            baseline_results = create_results_list(
                y_true_np, predictions, baseline_decisions, metadata_list
            )
            baseline_fairness = compute_fairness_from_results(baseline_results)

        # Sweep confidence thresholds (with cost_alpha=1.0 so Strategy B never triggers)
        confidence_results = sweep_confidence_thresholds(
            y_true,
            y_probs,
            confidence_thresholds,
            metadata_list=metadata_list,
            compute_fairness=include_fairness,
            cost_alpha_fixed=1.0,
        )

        # Sweep cost alphas (with confidence_threshold=0.0 so Strategy A always accepts AI)
        cost_results = sweep_cost_alphas(
            y_true,
            y_probs,
            cost_alphas,
            metadata_list=metadata_list,
            compute_fairness=include_fairness,
            confidence_threshold_fixed=0.0,
        )

        # Store results
        result_entry = {
            "model": model_name,
            "mode": mode,
            "num_classes": num_classes,
            "sweeps": {"confidence_threshold": confidence_results},
        }
        if baseline_fairness is not None:
            result_entry["baseline_fairness"] = baseline_fairness

        confidence_sweep_results[checkpoint_key] = result_entry

        cost_entry = {
            "model": model_name,
            "mode": mode,
            "num_classes": num_classes,
            "sweeps": {"cost_alpha": cost_results},
        }
        if baseline_fairness is not None:
            cost_entry["baseline_fairness"] = baseline_fairness

        cost_sweep_results[checkpoint_key] = cost_entry

        print(f"Sweep complete for {checkpoint_key}")

    # Save results
    os.makedirs("experiments", exist_ok=True)

    with open(CONFIDENCE_SWEEP_FILE, "w") as f:
        yaml.dump(
            confidence_sweep_results, f, default_flow_style=False, sort_keys=False
        )
    print(f"\nConfidence threshold sweep saved to {CONFIDENCE_SWEEP_FILE}")

    with open(COST_SWEEP_FILE, "w") as f:
        yaml.dump(cost_sweep_results, f, default_flow_style=False, sort_keys=False)
    print(f"Cost alpha sweep saved to {COST_SWEEP_FILE}")

    print("\n" + "=" * 60)
    print("All sweeps complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
