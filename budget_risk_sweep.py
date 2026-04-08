"""
budget_risk_sweep.py
--------------------
Sweep over cost values for budget-risk collaboration strategy.

This script performs a grid search over cost parameters:
- For each cost value: run budget analysis (10%, 20%, ..., 100%)
- Calculate metrics and fairness at each point

Usage:
    python budget_risk_sweep.py
    python budget_risk_sweep.py --cost-start 0.001 --cost-end 1.0 --cost-step 0.2
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
DEFAULT_OUTPUT_FILE = "experiments/budget_risk_sweep.yaml"
DEFAULT_CONFIG = "configs/dataset_config.yaml"

DEFAULT_COST_START = 0.001
DEFAULT_COST_END = 1.0
DEFAULT_COST_STEP = 0.2

DEFAULT_BUDGET_START = 0.1
DEFAULT_BUDGET_END = 1.0
DEFAULT_BUDGET_STEP = 0.1


# --- RISK-BASED COLLABORATION ---


def calculate_risk(y_probs, cost=1.0):
    """
    Calculate risk score for each sample.
    Risk = P(abnormal) × cost
    """
    prob_abnormal = 1 - y_probs[:, 0]
    risk = prob_abnormal * cost
    return risk


def apply_budget_strategy(y_true, y_probs, budget, cost=1.0):
    """Apply budget-based collaboration strategy."""
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()

    risk = calculate_risk(y_probs, cost)
    risk_np = risk.detach().cpu().numpy()

    sorted_indices = np.argsort(risk_np)[::-1]

    n_samples = len(y_true_np)
    n_human = int(n_samples * budget)

    predictions = np.zeros(n_samples, dtype=int)
    decisions = ["AI"] * n_samples

    human_indices = sorted_indices[:n_human]
    for idx in human_indices:
        predictions[idx] = y_true_np[idx]
        decisions[idx] = "HUMAN"

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
    parser = argparse.ArgumentParser(description="Budget-Risk Sweep Analysis")
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
        "--cost-start",
        type=float,
        default=DEFAULT_COST_START,
        help="Cost sweep start (default: 0.001)",
    )
    parser.add_argument(
        "--cost-end",
        type=float,
        default=DEFAULT_COST_END,
        help="Cost sweep end (default: 1.0)",
    )
    parser.add_argument(
        "--cost-step",
        type=float,
        default=DEFAULT_COST_STEP,
        help="Cost sweep step (default: 0.2)",
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
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        default=False,
        help="Compute fairness metrics (by gender/age) at each sweep point",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Budget-Risk Sweep Analysis")
    print("=" * 60)

    checkpoint_dir = args.checkpoint_dir
    dataset = args.dataset
    test_dir = f"data/{dataset}/test"
    output_file = args.output
    modes_to_test = args.modes
    use_config_subjects = args.config_subjects
    include_fairness = args.include_fairness

    # Generate cost levels
    cost_levels = np.arange(
        args.cost_start,
        args.cost_end + args.cost_step,
        args.cost_step,
    ).tolist()
    cost_levels = [round(c, 4) for c in cost_levels]

    # Generate budget levels
    budget_levels = np.arange(
        args.budget_start,
        args.budget_end + args.budget_step,
        args.budget_step,
    ).tolist()
    budget_levels = [round(b, 2) for b in budget_levels]

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Cost range: {cost_levels}")
    print(f"Budget range: {budget_levels}")
    print(f"Include fairness: {include_fairness}")

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

            if include_fairness:
                test_dataset = EEGCWTMetadataDataset(
                    base_dataset, "data/nmt_metadata.csv"
                )
            else:
                test_dataset = base_dataset

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Run inference
        if include_fairness:
            y_true, y_probs, metadata_list = run_inference_with_metadata(
                model, test_loader, device
            )
        else:
            y_true, y_probs = run_inference_with_metadata(model, test_loader, device)[
                :2
            ]
            metadata_list = None

        print(f"Running budget-risk sweep...")

        y_true_np = y_true.detach().cpu().numpy()

        # Cost sweep
        cost_sweep_results = {}
        for cost in cost_levels:
            print(f"  Cost: {cost}")

            # Calculate baseline for this cost
            if include_fairness and metadata_list is not None:
                baseline_predictions = (
                    torch.argmax(y_probs, dim=1).detach().cpu().numpy()
                )
                baseline_decisions = ["AI"] * len(baseline_predictions)
                baseline_results = create_results_list(
                    y_true_np, baseline_predictions, baseline_decisions, metadata_list
                )
                baseline_fairness = compute_fairness_from_results(baseline_results)
            else:
                baseline_fairness = None

            # Budget sweep for this cost
            budget_results = {}
            for budget in budget_levels:
                predictions, decisions = apply_budget_strategy(
                    y_true, y_probs, budget, cost
                )

                # Compute metrics
                metrics = compute_metrics(y_true_np, predictions, decisions)

                sweep_result = {"metrics": metrics}

                # Compute fairness if requested
                if include_fairness and metadata_list is not None:
                    results_list = create_results_list(
                        y_true_np, predictions, decisions, metadata_list
                    )
                    fairness = compute_fairness_from_results(results_list)
                    sweep_result["fairness"] = fairness

                budget_results[budget] = sweep_result

            cost_sweep_results[cost] = {
                "cost": cost,
                "baseline_fairness": baseline_fairness,
                "results": budget_results,
            }

        # Store results
        results[checkpoint_key] = {
            "model": model_name,
            "mode": mode,
            "num_classes": num_classes,
            "cost_sweep": cost_sweep_results,
        }

        print(f"Budget-risk sweep complete for {checkpoint_key}")

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(
            results, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\nResults saved to {output_file}")
    print("\n" + "=" * 60)
    print("Budget-Risk Sweep complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
