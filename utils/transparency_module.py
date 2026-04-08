"""
transparency_module.py
---------------------
Transparency and interpretability functions for EEG classification.

Provides methods to visualize which time-frequency regions contribute to:
1. Class difference attribution (Abnormal vs Normal)
2. Model uncertainty attribution

Usage:
    from utils.transparency_module import (
        generate_transparency_report,
        compute_class_difference_attribution,
        compute_uncertainty_attribution,
    )
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from sklearn.metrics import classification_report


def get_uncertain_samples_indices(y_probs, confidence_threshold=0.5):
    """
    Get indices of samples where model is uncertain.

    A sample is considered uncertain if max probability < confidence_threshold.

    Parameters
    ----------
    y_probs : torch.Tensor or np.ndarray
        Model predictions (after softmax), shape [N, num_classes]
    confidence_threshold : float
        Maximum confidence below which sample is considered uncertain

    Returns
    -------
    uncertain_indices : list
        Indices of uncertain samples
    """
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    max_probs = np.max(y_probs, axis=1)
    uncertain_mask = max_probs < confidence_threshold
    uncertain_indices = np.where(uncertain_mask)[0].tolist()

    return uncertain_indices


def get_abnormal_samples_indices(y_probs, threshold=0.5):
    """
    Get indices of samples predicted as abnormal.

    Abnormal = Slowing Waves (class 1) + Spike/Sharp Waves (class 2)

    Parameters
    ----------
    y_probs : torch.Tensor or np.ndarray
        Model predictions, shape [N, num_classes]
        Class 0 = Normal, Class 1 = Slowing Waves, Class 2 = Spike/Sharp
    threshold : float
        Minimum probability to consider a class as predicted

    Returns
    -------
    abnormal_indices : list
        Indices of samples predicted as abnormal
    """
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    # P(abnormal) = P(class 1) + P(class 2)
    prob_abnormal = y_probs[:, 1] + y_probs[:, 2]
    abnormal_mask = prob_abnormal > threshold
    abnormal_indices = np.where(abnormal_mask)[0].tolist()

    return abnormal_indices


def compute_class_difference_attribution(model, input_tensor, num_steps=50):
    """
    Compute class difference attribution: Abnormal - Normal.

    This shows which regions contribute to classifying a sample as abnormal
    rather than normal. Useful for identifying spikes/slowing patterns.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier
    input_tensor : torch.Tensor
        Input image, shape [1, C, H, W] or [C, H, W]
    num_steps : int
        Number of steps for integrated gradients

    Returns
    -------
    attribution : torch.Tensor
        Attribution scores, same shape as input
    delta : float
        Convergence delta (verification score)
    """
    model.eval()

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    ig = IntegratedGradients(model)

    # Get baseline (zero image)
    baseline = torch.zeros_like(input_tensor)

    # Compute attribution for abnormal classes (1 and 2)
    # We'll compute for class 1 and class 2, then average

    attr_class1, delta1 = ig.attribute(
        input_tensor,
        baseline=baseline,
        target=1,  # Slowing Waves
        return_convergence_delta=True,
    )

    attr_class2, delta2 = ig.attribute(
        input_tensor,
        baseline=baseline,
        target=2,  # Spike/Sharp Waves
        return_convergence_delta=True,
    )

    # Average attribution for both abnormal classes
    attribution = (attr_class1 + attr_class2) / 2
    delta = (delta1 + delta2) / 2

    return attribution, delta


def compute_uncertainty_attribution(model, input_tensor, num_steps=50):
    """
    Compute uncertainty attribution using gradient magnitude.

    This shows which regions the model is most uncertain about.
    High gradient magnitude = model is uncertain about these regions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier
    input_tensor : torch.Tensor
        Input image, shape [1, C, H, W] or [C, H, W]
    num_steps : int
        Number of steps for integrated gradients

    Returns
    -------
    attribution : torch.Tensor
        Attribution based on gradient magnitude (uncertainty)
    """
    model.eval()

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

    # Get max probability and its class
    max_prob, pred_class = torch.max(probs, dim=1)

    # Uncertainty = 1 - max_prob
    uncertainty = 1 - max_prob

    # Backpropagate to get gradients
    model.zero_grad()
    uncertainty.backward(retain_graph=True)

    # Gradient magnitude = uncertainty attribution
    attribution = torch.abs(input_tensor.grad)

    return attribution


def create_visualization(
    input_tensor, attribution, save_path, title="AI Attention Map"
):
    """
    Create and save heatmap visualization.

    Parameters
    ----------
    input_tensor : torch.Tensor or np.ndarray
        Input image
    attribution : torch.Tensor or np.ndarray
        Attribution scores
    save_path : str
        Path to save visualization
    title : str
        Title for the visualization
    """
    # Convert to numpy if needed
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.detach().cpu().numpy()
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()

    # Handle shapes
    if input_tensor.ndim == 4:
        input_tensor = input_tensor.squeeze()
    if attribution.ndim == 4:
        attribution = attribution.squeeze()

    # Transpose from (C, H, W) to (H, W, C)
    if input_tensor.shape[0] == 3:
        input_tensor = np.transpose(input_tensor, (1, 2, 0))
        attribution = np.transpose(attribution, (1, 2, 0))

    # Normalize for visualization
    input_tensor = (input_tensor - input_tensor.min()) / (
        input_tensor.max() - input_tensor.min() + 1e-8
    )
    attribution = np.abs(attribution).sum(axis=-1)  # Sum across channels

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(input_tensor)
    axes[0].set_title("Original Scalogram")
    axes[0].axis("off")

    # Attribution heatmap overlay
    im = axes[1].imshow(attribution, cmap="hot", alpha=0.7)
    axes[1].imshow(input_tensor, alpha=0.3)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def create_comparison_visualization(
    input_tensor, class_diff_attr, uncertainty_attr, save_path, sample_info=None
):
    """
    Create side-by-side comparison of both attribution methods.

    Parameters
    ----------
    input_tensor : torch.Tensor or np.ndarray
        Input image
    class_diff_attr : torch.Tensor
        Class difference attribution
    uncertainty_attr : torch.Tensor
        Uncertainty attribution
    save_path : str
        Path to save visualization
    sample_info : dict, optional
        Metadata about the sample
    """
    # Convert to numpy
    for arr in [input_tensor, class_diff_attr, uncertainty_attr]:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

    if input_tensor.ndim == 4:
        input_tensor = input_tensor.squeeze()
    if class_diff_attr.ndim == 4:
        class_diff_attr = class_diff_attr.squeeze()
    if uncertainty_attr.ndim == 4:
        uncertainty_attr = uncertainty_attr.squeeze()

    # Transpose from (C, H, W) to (H, W, C)
    if input_tensor.shape[0] == 3:
        input_tensor = np.transpose(input_tensor, (1, 2, 0))
        class_diff_attr = np.transpose(class_diff_attr, (1, 2, 0))
        uncertainty_attr = np.transpose(uncertainty_attr, (1, 2, 0))

    # Normalize
    input_tensor = (input_tensor - input_tensor.min()) / (
        input_tensor.max() - input_tensor.min() + 1e-8
    )
    class_diff_attr = np.abs(class_diff_attr).sum(axis=-1)
    uncertainty_attr = np.abs(uncertainty_attr).sum(axis=-1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Row 1: Original and Class Difference
    axes[0, 0].imshow(input_tensor)
    axes[0, 0].set_title("Original Scalogram")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(class_diff_attr, cmap="hot", alpha=0.7)
    axes[0, 1].imshow(input_tensor, alpha=0.3)
    axes[0, 1].set_title("Class Difference\n(Abnormal vs Normal)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Row 2: Original and Uncertainty
    axes[1, 0].imshow(input_tensor)
    axes[1, 0].set_title("Original Scalogram")
    axes[1, 0].axis("off")

    im2 = axes[1, 1].imshow(uncertainty_attr, cmap="hot", alpha=0.7)
    axes[1, 1].imshow(input_tensor, alpha=0.3)
    axes[1, 1].set_title("Uncertainty Attribution\n(Model Confusion)")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    # Add sample info if provided
    if sample_info:
        info_text = f"Confidence: {sample_info.get('confidence', 'N/A'):.3f}\n"
        info_text += f"Prediction: {sample_info.get('prediction', 'N/A')}\n"
        info_text += f"True Label: {sample_info.get('true_label', 'N/A')}"
        fig.text(
            0.5,
            0.02,
            info_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def generate_transparency_report(
    model,
    test_loader,
    confidence_threshold=0.5,
    num_samples=15,
    methods=["class_diff", "uncertainty"],
    output_dir="experiments/transparency/",
    device="cpu",
):
    """
    Generate transparency report for uncertain samples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier
    test_loader : DataLoader
        Test data loader (should return images, labels, metadata)
    confidence_threshold : float
        Threshold below which samples are considered uncertain
    num_samples : int
        Maximum number of samples to visualize
    methods : list
        Attribution methods to use: "class_diff", "uncertainty", or both
    output_dir : str
        Directory to save visualizations
    device : str
        Device to run inference on

    Returns
    -------
    metadata : dict
        Information about generated visualizations
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Collect predictions and metadata
    all_images = []
    all_labels = []
    all_probs = []
    all_metadata = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = None

            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)

            all_images.append(images.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

            if metadata:
                # Handle dict-of-lists format
                if isinstance(metadata, dict):
                    for j in range(images.shape[0]):
                        all_metadata.append(
                            {
                                "subject_id": str(metadata["subject_id"][j])
                                if hasattr(metadata["subject_id"][j], "item")
                                else str(metadata["subject_id"][j]),
                                "gender": str(metadata["gender"][j])
                                if hasattr(metadata["gender"][j], "item")
                                else str(metadata["gender"][j]),
                                "age": int(metadata["age"][j].item())
                                if hasattr(metadata["age"][j], "item")
                                else int(metadata["age"][j]),
                            }
                        )
                else:
                    all_metadata.extend(metadata)

    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # Get uncertain sample indices
    uncertain_indices = get_uncertain_samples_indices(all_probs, confidence_threshold)

    print(
        f"Found {len(uncertain_indices)} uncertain samples (confidence < {confidence_threshold})"
    )

    # Limit number of samples
    if len(uncertain_indices) > num_samples:
        uncertain_indices = uncertain_indices[:num_samples]

    print(f"Generating visualizations for {len(uncertain_indices)} samples...")

    # Create subdirectories
    class_diff_dir = os.path.join(output_dir, "class_difference")
    uncertainty_dir = os.path.join(output_dir, "uncertainty")
    comparison_dir = os.path.join(output_dir, "comparison")

    os.makedirs(class_diff_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Generate visualizations
    metadata = {
        "confidence_threshold": confidence_threshold,
        "num_samples_requested": num_samples,
        "num_samples_generated": len(uncertain_indices),
        "samples": [],
    }

    for i, idx in enumerate(uncertain_indices):
        input_tensor = all_images[idx]
        true_label = all_labels[idx].item()
        probs = all_probs[idx]
        max_prob, pred_class = torch.max(probs, dim=0)
        max_prob = max_prob.item()
        pred_class = pred_class.item()

        sample_info = {
            "index": idx,
            "true_label": int(true_label),
            "prediction": int(pred_class),
            "confidence": max_prob,
        }

        if all_metadata and idx < len(all_metadata):
            sample_info.update(all_metadata[idx])

        # Compute attributions
        input_for_model = input_tensor.unsqueeze(0).to(device)

        # Class difference attribution
        if "class_diff" in methods:
            class_diff_attr, _ = compute_class_difference_attribution(
                model, input_for_model
            )
            class_diff_attr = class_diff_attr.squeeze().cpu()

            save_path = os.path.join(
                class_diff_dir, f"sample_{i + 1:03d}_uncertain.png"
            )
            create_visualization(
                input_tensor,
                class_diff_attr,
                save_path,
                title=f"Abnormal Attribution (Conf: {max_prob:.3f})",
            )

        # Uncertainty attribution
        if "uncertainty" in methods:
            uncertainty_attr = compute_uncertainty_attribution(model, input_for_model)
            uncertainty_attr = uncertainty_attr.squeeze().cpu()

            save_path = os.path.join(
                uncertainty_dir, f"sample_{i + 1:03d}_uncertain.png"
            )
            create_visualization(
                input_tensor,
                uncertainty_attr,
                save_path,
                title=f"Uncertainty Regions (Conf: {max_prob:.3f})",
            )

        # Comparison visualization
        if "class_diff" in methods and "uncertainty" in methods:
            save_path = os.path.join(comparison_dir, f"sample_{i + 1:03d}_both.png")
            create_comparison_visualization(
                input_tensor,
                class_diff_attr,
                uncertainty_attr,
                save_path,
                sample_info=sample_info,
            )

        metadata["samples"].append(sample_info)
        print(
            f"  Processed sample {i + 1}/{len(uncertain_indices)}: idx={idx}, conf={max_prob:.3f}, pred={pred_class}"
        )

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTransparency report saved to {output_dir}")
    print(
        f"  - Class difference: {len([m for m in methods if 'class_diff' in m])} visualizations"
    )
    print(
        f"  - Uncertainty: {len([m for m in methods if 'uncertainty' in m])} visualizations"
    )
    print(f"  - Comparison: {'both' in methods} visualization")
    print(f"  - Metadata: {metadata_path}")

    return metadata


# --- MAIN FUNCTION FOR CLI ---


def main():
    """CLI entry point for generating transparency reports."""
    import argparse
    from torch.utils.data import DataLoader
    from train import EEGCWTDataset, create_model, get_default_transforms
    from utils.dataset_with_metadata import EEGCWTMetadataDataset

    parser = argparse.ArgumentParser(description="Generate transparency heatmaps")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Uncertainty threshold"
    )
    parser.add_argument(
        "--num-samples", type=int, default=15, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["class_diff", "uncertainty", "both"],
        help="Attribution methods",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/transparency/",
        help="Output directory",
    )
    parser.add_argument("--dataset", type=str, default="nmt", help="Dataset name")
    parser.add_argument(
        "--mode", type=str, default="three_class", help="Mode: three_class or binary"
    )
    parser.add_argument(
        "--config-subjects",
        action="store_true",
        default=False,
        help="Use config-based test subjects",
    )

    args = parser.parse_args()

    methods = (
        ["class_diff", "uncertainty"] if args.methods == "both" else [args.methods]
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.checkpoint.split("/")[-1].replace("_best.pt", "").split("_")[1],
        num_classes=3,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test data
    transforms = get_default_transforms()
    test_dir = f"data/{args.dataset}/test"
    base_dataset = EEGCWTDataset(test_dir, mode=args.mode, transform=transforms["test"])
    test_dataset = EEGCWTMetadataDataset(base_dataset, "data/nmt_metadata.csv")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Generate report
    generate_transparency_report(
        model,
        test_loader,
        confidence_threshold=args.confidence,
        num_samples=args.num_samples,
        methods=methods,
        output_dir=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()
