import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import EEGCWTDataset, create_model, get_default_transforms
from utils.dataset_with_metadata import EEGCWTMetadataDataset, classify_age_group


DEFAULT_DATASET = "nmt"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_METADATA_CSV = "data/nmt_metadata.csv"
DEFAULT_CONFIG = "configs/dataset_config.yaml"


def get_config_split_subjects(config_path: str = DEFAULT_CONFIG, dataset: str = "nmt"):
    """Read config file to get subjects in train/valid/test splits."""
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

    return train_ids, valid_ids, test_ids


def get_test_subjects_from_config(
    config_path: str = DEFAULT_CONFIG, dataset: str = "nmt"
):
    """
    Get test subjects from config file.
    Uses abnormal EDF subjects NOT in train/valid splits + includes overlapping from test config.
    """
    train_ids, valid_ids, test_ids = get_config_split_subjects(config_path, dataset)

    # Get all abnormal EDF subjects with metadata
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

    # Include overlapping subjects from test config
    overlapping = abnormal_subjects & test_ids

    final_test_subjects = available_for_test | overlapping

    return final_test_subjects


def apply_strategy_a(y_true, y_probs, confidence_threshold, metadata_list):
    confidences, predictions = torch.max(y_probs, dim=1)

    results = []
    for i in range(len(predictions)):
        decision = "AI" if confidences[i] >= confidence_threshold else "HUMAN"
        final_label = predictions[i].item() if decision == "AI" else y_true[i].item()

        meta = metadata_list[i]
        gender = meta.get("gender", "unknown")
        if hasattr(gender, "item"):
            gender = gender.item()

        age = meta.get("age", -1)
        if hasattr(age, "item"):
            age = age.item()

        results.append(
            {
                "y_true": int(y_true[i].item()),
                "y_pred": int(final_label),
                "confidence": float(confidences[i].item()),
                "decision": decision,
                "subject_id": str(meta.get("subject_id", "unknown")),
                "gender": str(gender),
                "age": int(age),
            }
        )

    return results


def apply_strategy_b(y_true, y_probs, cost_alpha, metadata_list):
    results = []

    for i in range(len(y_probs)):
        prob_pathology = 1 - y_probs[i][0]

        decision = "HUMAN" if prob_pathology > cost_alpha else "AI"

        _, predictions = torch.max(y_probs, dim=1)
        final_label = predictions[i].item() if decision == "AI" else y_true[i].item()

        meta = metadata_list[i]
        gender = meta.get("gender", "unknown")
        if hasattr(gender, "item"):
            gender = gender.item()

        age = meta.get("age", -1)
        if hasattr(age, "item"):
            age = age.item()

        results.append(
            {
                "y_true": int(y_true[i].item()),
                "y_pred": int(final_label),
                "prob_pathology": float(prob_pathology.item()),
                "decision": decision,
                "subject_id": str(meta.get("subject_id", "unknown")),
                "gender": str(gender),
                "age": int(age),
            }
        )

    return results


def compute_group_metrics(df, group_key):
    age_groups = df["age"].apply(classify_age_group)
    df = df.assign(age_group=age_groups)

    group_metrics = {}

    for group_name, group_df in df.groupby(group_key):
        group_key_str = str(group_name)

        y_true = group_df["y_true"].values
        y_pred = group_df["y_pred"].values
        decisions = group_df["decision"].values

        report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        human_count = (decisions == "HUMAN").sum()
        escalation_rate = (
            float(human_count / len(decisions)) * 100 if len(decisions) > 0 else 0
        )

        group_metrics[group_key_str] = {
            "accuracy": round(float(report["accuracy"]), 4),
            "precision": round(float(report["weighted avg"]["precision"]), 4),
            "recall": round(float(report["weighted avg"]["recall"]), 4),
            "f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "escalation_rate": round(escalation_rate, 2),
            "sample_count": int(len(group_df)),
        }

    return group_metrics


def analyze_fairness(results_list):
    if isinstance(results_list, list):
        results_df = pd.DataFrame(results_list)
    else:
        results_df = results_list.copy()

    debug_info = {
        "total_samples": len(results_df),
        "unique_genders": results_df["gender"].unique().tolist(),
        "unique_ages": sorted(results_df["age"].unique().tolist()),
    }

    gender_metrics = compute_group_metrics(results_df, "gender")

    results_df["age_group"] = results_df["age"].apply(classify_age_group)
    age_metrics = compute_group_metrics(results_df, "age_group")

    y_true = results_df["y_true"].values
    y_pred = results_df["y_pred"].values
    decisions = results_df["decision"].values

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    human_count = (decisions == "HUMAN").sum()
    overall = {
        "accuracy": round(float(report["accuracy"]), 4),
        "precision": round(float(report["weighted avg"]["precision"]), 4),
        "recall": round(float(report["weighted avg"]["recall"]), 4),
        "f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "escalation_rate": round(float(human_count / len(decisions)) * 100, 2),
        "sample_count": len(results_df),
    }

    return {
        "overall": overall,
        "by_gender": gender_metrics,
        "by_age_group": age_metrics,
        "debug": debug_info,
    }


def find_checkpoints(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, f"{DEFAULT_DATASET}_*_best.pt")
    files = glob.glob(pattern)
    return sorted(files)


def parse_checkpoint_filename(checkpoint_path, dataset):
    basename = os.path.basename(checkpoint_path)
    parts = basename.replace(".pt", "").split("_")

    if len(parts) >= 3:
        model_name = parts[1]
        mode = parts[2]
        num_classes = 3 if "three" in mode else 2
        return model_name, mode, num_classes

    return None, None, None


def run_inference_with_metadata(model, dataloader, device):
    model.eval()

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

            all_labels.append(labels)
            all_probs.append(probs)

            batch_size = labels.shape[0]
            for j in range(batch_size):
                subject_id = metadata["subject_id"][j]
                if hasattr(subject_id, "item"):
                    subject_id = subject_id.item()

                gender = metadata["gender"][j]
                if hasattr(gender, "item"):
                    gender = gender.item()

                age = metadata["age"][j]
                if hasattr(age, "item"):
                    age = age.item()

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


def main():
    parser = argparse.ArgumentParser(description="EEG Collaboration Fairness Analysis")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--metadata-csv", type=str, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--cost-alpha", type=float, default=0.15)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--debug", action="store_true", help="Print debug info")
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
    debug = args.debug
    use_config_subjects = args.config_subjects
    modes_to_test = args.modes

    print(f"Dataset: {dataset}")
    print(f"Test dir: {test_dir}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Use config subjects: {use_config_subjects}")

    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata CSV not found at {metadata_csv}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms_dict = get_default_transforms()

    if args.model and args.mode:
        checkpoint_filename = f"{dataset}_{args.model}_{args.mode}_best.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return

        checkpoint_files = [checkpoint_path]
    else:
        checkpoint_files = find_checkpoints(checkpoint_dir)

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s)")

    for checkpoint_path in checkpoint_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(checkpoint_path)}")

        model_name, mode, num_classes = parse_checkpoint_filename(
            checkpoint_path, dataset
        )

        if model_name is None:
            print(f"Skipping - could not parse model/mode")
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

        try:
            model = create_model(model_name, num_classes=num_classes).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"Loaded model")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        try:
            # Get test subjects from config if flag is True
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
            test_dataset = EEGCWTMetadataDataset(base_dataset, metadata_csv)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            print(
                f"Loaded test data: {len(test_dataset)} samples"
                + (
                    f" from {len(test_subject_ids)} subjects"
                    if test_subject_ids
                    else ""
                )
            )
        except Exception as e:
            print(f"Failed to load test data: {e}")
            continue

        y_true, y_probs, metadata_list = run_inference_with_metadata(
            model, test_loader, device
        )

        results_a = apply_strategy_a(
            y_true, y_probs, confidence_threshold, metadata_list
        )
        fairness_a = analyze_fairness(results_a)

        if debug:
            print(f"\nDEBUG Strategy A:")
            print(f"  Total samples: {fairness_a['debug']['total_samples']}")
            print(f"  Unique genders: {fairness_a['debug']['unique_genders']}")
            print(f"  Unique ages: {fairness_a['debug']['unique_ages']}")
            print(f"  By gender: {list(fairness_a['by_gender'].keys())}")
            print(f"  By age_group: {list(fairness_a['by_age_group'].keys())}")

        results_b = apply_strategy_b(y_true, y_probs, cost_alpha, metadata_list)
        fairness_b = analyze_fairness(results_b)

        if debug:
            print(f"\nDEBUG Strategy B:")
            print(f"  Total samples: {fairness_b['debug']['total_samples']}")
            print(f"  Unique genders: {fairness_b['debug']['unique_genders']}")
            print(f"  Unique ages: {fairness_b['debug']['unique_ages']}")
            print(f"  By gender: {list(fairness_b['by_gender'].keys())}")
            print(f"  By age_group: {list(fairness_b['by_age_group'].keys())}")

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

        output_dir = "experiments"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"fairness_{checkpoint_key}.yaml")

        with open(output_file, "w") as f:
            yaml.dump(
                output_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        print(f"Fairness analysis saved to {output_file}")

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


if __name__ == "__main__":
    main()
