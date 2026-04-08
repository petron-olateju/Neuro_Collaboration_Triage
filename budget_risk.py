"""
budget_risk.py
--------------
Budget-Risk based collaboration strategy evaluation.

This script evaluates a risk-based resource allocation approach:
1. Calculate risk = P(abnormal) × cost (where abnormal = Slowing Waves + Spike/Sharp waves)
2. Sort cases by risk (highest first)
3. For each budget level (10%, 20%, ..., 100%):
   - Send top budget% of highest-risk cases to human
   - AI handles the rest
   - Calculate overall metrics AND fairness (by gender/age)

Usage:
    python budget_risk.py
    python budget_risk.py --cost 2.0 --budget-step 0.05
"""

import os
import glob
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import EEGCWTDataset, create_model, get_default_transforms, load_config
from utils.dataset_with_metadata import EEGCWTMetadataDataset, classify_age_group
from test_and_collab_sweep import (
    create_results_list,
    compute_fairness_from_results,
    get_test_subjects_from_config,
)


# --- CONFIGURATION ---
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_DATASET = "nmt"
DEFAULT_OUTPUT_FILE = "experiments/budget_risk.yaml"
DEFAULT_CONFIG = "configs/dataset_config.yaml"

DEFAULT_BUDGET_START = 0.1
DEFAULT_BUDGET_END = 1.0
DEFAULT_BUDGET_STEP = 0.1
DEFAULT_COST = 1.0


# --- RISK-BASED COLLABORATION ---


def calculate_risk(y_probs, cost=1.0):
    """
    Calculate risk score for each sample.

    Risk = P(abnormal) × cost
    where P(abnormal) = P(Slowing Waves) + P(Spike/Sharp Waves) = 1 - P(Normal)

    Parameters
    ----------
    y_probs : torch.Tensor
        Tensor of shape [N, num_classes] with softmax probabilities
        Class 0 = Normal, Class 1 = Slowing Waves, Class 2 = Spike/Sharp Waves
    cost : float
        Cost multiplier for missing abnormal cases

    Returns
    -------
    risk : torch.Tensor
        Risk score for each sample
    """
    # P(abnormal) = 1 - P(Normal) = 1 - P(class 0)
    prob_abnormal = 1 - y_probs[:, 0]

    # Risk = P(abnormal) × cost
    risk = prob_abnormal * cost

    return risk


def apply_budget_strategy(y_true, y_probs, budget, cost=1.0):
    """
    Apply budget-based collaboration strategy.

    Sorts cases by risk (highest first) and allocates budget% to human review.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels
    y_probs : torch.Tensor
        Model softmax probabilities [N, num_classes]
    budget : float
        Fraction of cases to send to human (0.0 to 1.0)
    cost : float
        Cost parameter for risk calculation

    Returns
    -------
    predictions : np.ndarray
        Final predictions after collaboration
    decisions : list
        "HUMAN" or "AI" for each sample
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()

    # Calculate risk scores
    risk = calculate_risk(y_probs, cost)
    risk_np = risk.detach().cpu().numpy()

    # Sort by risk (highest first)
    sorted_indices = np.argsort(risk_np)[::-1]

    # Determine how many to send to human
    n_samples = len(y_true_np)
    n_human = int(n_samples * budget)

    # Make predictions based on budget allocation
    predictions = np.zeros(n_samples, dtype=int)
    decisions = ["AI"] * n_samples

    # Top risk cases go to human
    human_indices = sorted_indices[:n_human]
    for idx in human_indices:
        predictions[idx] = y_true_np[idx]  # Human gets the ground truth (oracle)
        decisions[idx] = "HUMAN"

    # Rest handled by AI
    ai_indices = sorted_indices[n_human:]
    ai_probs = y_probs_np[ai_indices]
    ai_predictions = np.argmax(ai_probs, axis=1)
    for i, idx in enumerate(ai_indices):
        predictions[idx] = ai_predictions[i]
        decisions[idx] = "AI"

    return predictions, decisions


def compute_metrics(y_true, y_pred, decisions=None):
    """Compute classification metrics."""
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(report["accuracy"]), 4),
        "precision": round(float(report["weighted avg"]["precision"]), 4),
        "recall": round(float(report["weighted avg"]["recall"]), 4),
        "f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }

    if decisions is not None:
        human_count = decisions.count("HUMAN")
        metrics["escalation_rate"] = round((human_count / len(decisions)) * 100, 2)

    return metrics


# --- INFERENCE ---


def run_inference_with_metadata(model, dataloader, device):
    """Run model inference and collect metadata."""
    model.eval()
    all_labels = []
    all_probs = []
    all_metadata = []

    with torch.no_grad():
        for batch in dataloader:
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


# --- CHECKPOINT PARSING ---


def find_checkpoints(checkpoint_dir):
    """Find all model checkpoints."""
    pattern = os.path.join(checkpoint_dir, f"{DEFAULT_DATASET}_*_best.pt")
    files = glob.glob(pattern)
    return sorted(files)


def parse_checkpoint_filename(filename, dataset):
    """Parse checkpoint filename to extract model, mode, and num_classes."""
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


# --- MAIN ---


def main():
    parser = argparse.ArgumentParser(description="Budget-Risk Collaboration Analysis")
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
        "--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output YAML file"
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=DEFAULT_COST,
        help="Cost parameter for risk calculation (default: 1.0)",
    )
    parser.add_argument(
        "--budget-start",
        type=float,
        default=DEFAULT_BUDGET_START,
        help="Budget sweep start (default: 0.1)",
    )
    parser.add_argument(
        "--budget-end",
        type=float,
        default=DEFAULT_BUDGET_END,
        help="Budget sweep end (default: 1.0)",
    )
    parser.add_argument(
        "--budget-step",
        type=float,
        default=DEFAULT_BUDGET_STEP,
        help="Budget sweep step (default: 0.1)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="three_class",
        choices=["binary", "three_class", "all"],
        help="Which modes to test: binary, three_class, or all (default: three_class)",
    )
    parser.add_argument(
        "--config-subjects",
        action="store_true",
        default=True,
        help="Use abnormal subjects from config for testing (default: True)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Budget-Risk Collaboration Analysis")
    print("=" * 60)

    checkpoint_dir = args.checkpoint_dir
    dataset = args.dataset
    test_dir = f"data/{dataset}/test"
    output_file = args.output
    cost = args.cost
    modes_to_test = args.modes
    use_config_subjects = args.config_subjects

    # Generate budget levels
    budget_levels = np.arange(
        args.budget_start,
        args.budget_end + args.budget_step,
        args.budget_step,
    ).tolist()
    budget_levels = [round(b, 2) for b in budget_levels]

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Cost parameter: {cost}")
    print(f"Budget levels: {budget_levels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms_dict = get_default_transforms()

    # Find checkpoints
    checkpoint_files = find_checkpoints(checkpoint_dir)
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    results = {}

    for checkpoint_path in checkpoint_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(checkpoint_path)}")

        # Parse checkpoint
        model_name, mode, num_classes = parse_checkpoint_filename(
            checkpoint_path, dataset
        )

        if model_name is None:
            print(f"Skipping - could not parse model/mode")
            continue

        # Skip modes based on --modes argument
        if modes_to_test == "three_class" and mode == "binary":
            print(f"Skipping {checkpoint_path} - binary mode skipped")
            continue
        if modes_to_test == "binary" and mode != "binary":
            print(f"Skipping {checkpoint_path} - three_class mode skipped")
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

            base_dataset = EEGCWTDataset(
                test_dir,
                mode=mode,
                transform=transforms_dict["test"],
                subject_ids=test_subject_ids,
            )

            # Wrap with metadata dataset for fairness
            test_dataset = EEGCWTMetadataDataset(base_dataset, "data/nmt_metadata.csv")

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Run inference
        y_true, y_probs, metadata_list = run_inference_with_metadata(
            model, test_loader, device
        )

        print(f"Running budget-risk analysis...")

        y_true_np = y_true.detach().cpu().numpy()

        # Calculate baseline (AI-only) for comparison
        baseline_predictions = torch.argmax(y_probs, dim=1).detach().cpu().numpy()
        baseline_decisions = ["AI"] * len(baseline_predictions)

        baseline_results = create_results_list(
            y_true_np, baseline_predictions, baseline_decisions, metadata_list
        )
        baseline_fairness = compute_fairness_from_results(baseline_results)

        # Run budget sweep
        budget_results = {}
        for budget in budget_levels:
            predictions, decisions = apply_budget_strategy(
                y_true, y_probs, budget, cost
            )

            # Compute metrics
            metrics = compute_metrics(y_true_np, predictions, decisions)

            # Compute fairness
            results_list = create_results_list(
                y_true_np, predictions, decisions, metadata_list
            )
            fairness = compute_fairness_from_results(results_list)

            budget_results[budget] = {
                "metrics": metrics,
                "fairness": fairness,
            }

        # Store results
        results[checkpoint_key] = {
            "model": model_name,
            "mode": mode,
            "num_classes": num_classes,
            "cost_parameter": cost,
            "baseline_fairness": baseline_fairness,
            "results": budget_results,
        }

        print(f"Budget-risk analysis complete for {checkpoint_key}")

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(
            results, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\nResults saved to {output_file}")
    print("\n" + "=" * 60)
    print("Budget-Risk analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
