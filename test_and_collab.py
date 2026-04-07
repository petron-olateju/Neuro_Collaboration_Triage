import os
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

# Import from train.py
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import EEGCWTDataset, create_model, get_default_transforms, load_config


# --- CONFIGURATION ---
EXPERIMENTS_FILE = "experiments/experiment.yaml"
DEFAULT_DATASET = "nmt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


# --- COLLABORATION LOGIC ---


def apply_collaboration_strategies(
    y_true, y_probs, confidence_threshold=0.85, cost_alpha=0.2
):
    """
    Compares AI-only vs Human-AI Collaboration.
    y_true: Ground truth labels (0=Normal)
    y_probs: Model softmax probabilities [N, num_classes]
    """
    results = {}

    confidences, predictions = torch.max(y_probs, dim=1)

    # --- BASELINE (AI ALONE) ---
    ai_alone_preds = predictions.detach().cpu().numpy()
    results["baseline"] = {
        "predictions": ai_alone_preds,
        "decisions": ["AI"] * len(predictions),
    }

    # --- STRATEGY A: Selective Prediction (Confidence-based) ---
    strat_a_decisions = []
    strat_a_final_labels = []

    for i in range(len(predictions)):
        if confidences[i] >= confidence_threshold:
            strat_a_decisions.append("AI")
            strat_a_final_labels.append(predictions[i].item())
        else:
            strat_a_decisions.append("HUMAN")
            strat_a_final_labels.append(y_true[i].item())

    results["strategy_a"] = {
        "labels": np.array(strat_a_final_labels),
        "decisions": strat_a_decisions,
    }

    # --- STRATEGY B: Cost-Aware Deferral (Risk-based) ---
    strat_b_decisions = []
    strat_b_final_labels = []

    for i in range(len(predictions)):
        prob_pathology = 1 - y_probs[i][0]

        if prob_pathology > cost_alpha:
            strat_b_decisions.append("HUMAN")
            strat_b_final_labels.append(y_true[i].item())
        else:
            strat_b_decisions.append("AI")
            strat_b_final_labels.append(predictions[i].item())

    results["strategy_b"] = {
        "labels": np.array(strat_b_final_labels),
        "decisions": strat_b_decisions,
    }

    return results, confidences, predictions


# --- EVALUATION AND METRICS ---


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


def print_collab_report(name, true, predicted, decisions):
    """Print formatted collaboration report."""
    print(f"\n=== {name} Report ===")

    human_count = decisions.count("HUMAN")
    esc_rate = (human_count / len(decisions)) * 100

    report = classification_report(true, predicted, zero_division=0)

    print(f"Human Escalation Rate: {esc_rate:.2f}%")
    print(f"System Performance (AI + Human Collaboration):")
    print(report)


# --- EXPERIMENT RECORDING ---


def load_experiments():
    """Load existing experiments or create new structure."""
    if os.path.exists(EXPERIMENTS_FILE):
        with open(EXPERIMENTS_FILE, "r") as f:
            return yaml.safe_load(f) or {"runs": {}}
    return {"runs": {}}


def save_experiments(experiments):
    """Save experiments to YAML file."""
    os.makedirs("experiments", exist_ok=True)
    with open(EXPERIMENTS_FILE, "w") as f:
        yaml.dump(experiments, f, default_flow_style=False, sort_keys=False)
    print(f"Experiments saved to {EXPERIMENTS_FILE}")


def record_experiment(
    timestamp,
    model_name,
    dataset,
    mode,
    num_classes,
    confidence_threshold,
    cost_alpha,
    results,
):
    """Record a single experiment run to the experiments file."""
    experiments = load_experiments()

    run_data = {
        "model": {
            "name": model_name,
            "dataset": dataset,
            "mode": mode,
            "num_classes": num_classes,
        },
        "collaboration": {
            "strategy_a": {"confidence_threshold": confidence_threshold},
            "strategy_b": {"cost_alpha": cost_alpha},
        },
        "results": {
            "baseline": results["baseline"],
            "strategy_a": results["strategy_a"],
            "strategy_b": results["strategy_b"],
        },
    }

    experiments["runs"][timestamp] = run_data
    save_experiments(experiments)


# --- MODEL INFERENCE ---


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


def find_checkpoints(checkpoint_dir):
    """Find all model checkpoints."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    return checkpoint_files


def parse_checkpoint_filename(filename, dataset):
    """Parse checkpoint filename to extract model and mode info."""
    basename = os.path.basename(filename)
    # Format: {dataset}_{model}_{mode}_best.pt
    # Example: nmt_vgg16_binary_best.pt

    # Remove _best.pt suffix
    name_without_ext = basename.replace("_best.pt", "")

    # Split by underscore
    parts = name_without_ext.split("_")

    if len(parts) >= 3:
        # Expected: [dataset, model, mode]
        # But dataset could have underscores, so be careful
        if parts[0] == dataset:
            model = parts[1]
            mode = parts[2]
            return model, mode

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="EEG Collaboration Analysis - Evaluate Human-AI collaboration strategies"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (default: nmt)",
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

    args = parser.parse_args()

    print("=" * 60)
    print("EEG Collaboration Analysis")
    print("=" * 60)

    config = load_config("configs/training.yaml")
    training_config = config.get("training", {})

    # Collaboration policy parameters from CLI
    confidence_threshold = args.confidence_threshold
    cost_alpha = args.cost_alpha

    dataset = args.dataset
    test_dir = f"data/{dataset}/test"
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Confidence threshold (Strategy A): {confidence_threshold}")
    print(f"Cost alpha (Strategy B): {cost_alpha}")

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

    # Process each checkpoint
    for checkpoint_path in checkpoint_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {checkpoint_path}")
        print("=" * 60)

        # Parse checkpoint name
        model_name, mode = parse_checkpoint_filename(checkpoint_path, dataset)

        if model_name is None:
            print(f"Skipping {checkpoint_path} - could not parse model/mode")
            continue

        print(f"Model: {model_name}, Mode: {mode}")

        # Determine num_classes
        num_classes = 3 if mode == "three_class" else 2

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
            test_dataset = EEGCWTDataset(
                test_dir, mode=mode, transform=transforms_dict["test"]
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )
            print(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        # Run inference
        y_true, y_probs = run_inference(model, test_loader, device)

        # Apply collaboration strategies
        results, confidences, predictions = apply_collaboration_strategies(
            y_true,
            y_probs,
            confidence_threshold=confidence_threshold,
            cost_alpha=cost_alpha,
        )

        # Compute metrics for each strategy
        y_true_np = y_true.detach().cpu().numpy()

        baseline_metrics = compute_metrics(
            y_true_np, results["baseline"]["predictions"]
        )
        results["baseline"] = baseline_metrics

        strategy_a_metrics = compute_metrics(
            y_true_np,
            results["strategy_a"]["labels"],
            results["strategy_a"]["decisions"],
        )
        results["strategy_a"] = strategy_a_metrics

        strategy_b_metrics = compute_metrics(
            y_true_np,
            results["strategy_b"]["labels"],
            results["strategy_b"]["decisions"],
        )
        results["strategy_b"] = strategy_b_metrics

        # Print reports
        print_collab_report(
            "BASELINE (AI ALONE)",
            y_true_np,
            results["baseline"]["predictions"],
            ["AI"] * len(y_true_np),
        )

        print_collab_report(
            "STRATEGY A (CONFIDENCE-BASED)",
            y_true_np,
            results["strategy_a"]["labels"],
            results["strategy_a"]["decisions"],
        )

        print_collab_report(
            "STRATEGY B (COST-AWARE TRIAGE)",
            y_true_np,
            results["strategy_b"]["labels"],
            results["strategy_b"]["decisions"],
        )

        # Record to experiments
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        record_experiment(
            timestamp=timestamp,
            model_name=model_name,
            dataset=dataset,
            mode=mode,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            cost_alpha=cost_alpha,
            results=results,
        )

        print(f"\n✓ Results recorded to experiments/experiment.yaml")

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
