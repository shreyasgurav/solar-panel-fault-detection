"""
evaluate.py - Evaluation & Metrics for Solar Panel Fault Detection Models

This module handles:
1. Model evaluation on the test set
2. Computing all required metrics: Accuracy, Precision, Recall, F1-Score
3. Generating confusion matrix heatmap
4. Producing detailed classification report
5. Comparing two models side-by-side

Framework: PyTorch + scikit-learn for metrics
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_predictions(model, data_loader, device):
    """
    Runs the model on the entire dataset and collects all predictions and true labels.

    Sets model to eval mode to disable dropout and freeze batch normalization,
    and uses torch.no_grad() to disable gradient computation for efficiency.

    Args:
        model (nn.Module): Trained model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run inference on.

    Returns:
        all_preds (numpy.ndarray): Predicted class indices.
        all_labels (numpy.ndarray): True class indices.
        all_probs (numpy.ndarray): Predicted probabilities for each class.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, class_names):
    """
    Computes comprehensive classification metrics.

    Metrics computed:
    - Accuracy: Overall correct predictions / total predictions
    - Precision (per class + macro): Of all predicted as class X, how many were actually X
    - Recall (per class + macro): Of all actual class X, how many were correctly predicted
    - F1-Score (per class + macro): Harmonic mean of precision and recall

    Macro averaging treats all classes equally regardless of their size,
    which is important for imbalanced datasets to ensure minority classes
    (like Physical-Damage) are weighted fairly in the overall metric.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        class_names (list): List of class name strings.

    Returns:
        metrics (dict): Dictionary containing all computed metrics.
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100

    # Per-class and macro-averaged metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Macro averages (unweighted mean across all classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted averages (weighted by class support — accounts for imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision_weighted': precision_weighted * 100,
        'recall_weighted': recall_weighted * 100,
        'f1_weighted': f1_weighted * 100,
    }

    return metrics


def print_metrics(metrics, class_names, model_name="Model"):
    """
    Prints all metrics in a formatted, readable table.

    Args:
        metrics (dict): Computed metrics from compute_metrics().
        class_names (list): Class name strings.
        model_name (str): Name of the model for the header.
    """
    print(f"\n{'='*70}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'='*70}")
    print(f"\n  Overall Accuracy: {metrics['accuracy']:.2f}%\n")

    # Per-class metrics table
    header = f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    print(f"  {header}")
    print(f"  {'-'*52}")

    for i, name in enumerate(class_names):
        p = metrics['precision_per_class'][i] * 100
        r = metrics['recall_per_class'][i] * 100
        f = metrics['f1_per_class'][i] * 100
        print(f"  {name:<22} {p:>9.2f}% {r:>9.2f}% {f:>9.2f}%")

    print(f"  {'-'*52}")
    print(f"  {'Macro Average':<22} {metrics['precision_macro']:>9.2f}% {metrics['recall_macro']:>9.2f}% {metrics['f1_macro']:>9.2f}%")
    print(f"  {'Weighted Average':<22} {metrics['precision_weighted']:>9.2f}% {metrics['recall_weighted']:>9.2f}% {metrics['f1_weighted']:>9.2f}%")
    print(f"{'='*70}\n")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", save_dir="outputs/plots"):
    """
    Generates and saves a confusion matrix heatmap using seaborn.

    The confusion matrix shows:
    - Diagonal elements: Correct predictions (true positives for each class)
    - Off-diagonal elements: Misclassifications (which classes get confused)

    This is critical for understanding model behavior — e.g., if Dusty and Clean
    are frequently confused, it suggests visual similarity between these classes.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        class_names (list): Class name strings for axis labels.
        model_name (str): Model name for title and filename.
        save_dir (str): Directory to save the plot.

    Returns:
        cm (numpy.ndarray): The confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    # Create a well-formatted heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in each cell
        fmt='d',              # Integer format
        cmap='Blues',          # Blue color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Rotate tick labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    return cm


def get_classification_report(y_true, y_pred, class_names):
    """
    Generates a detailed classification report using sklearn.

    The report includes precision, recall, F1-score, and support (number of
    true instances) for each class, along with macro/weighted averages.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        class_names (list): Class name strings.

    Returns:
        report (str): Formatted classification report string.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    return report


def evaluate_model(model, test_loader, class_names, device, model_name="Model", save_dir="outputs/plots"):
    """
    Complete evaluation pipeline for a single model.

    Steps:
    1. Get predictions on the test set
    2. Compute all metrics (accuracy, precision, recall, F1)
    3. Print detailed results
    4. Generate and save confusion matrix
    5. Print classification report

    Args:
        model (nn.Module): Trained model to evaluate.
        test_loader (DataLoader): Test set DataLoader.
        class_names (list): Class name strings.
        device (torch.device): Device for inference.
        model_name (str): Model name for labeling results.
        save_dir (str): Directory for saving plots.

    Returns:
        metrics (dict): All computed metrics.
        y_pred (numpy.ndarray): Predictions for further analysis.
        y_true (numpy.ndarray): True labels for further analysis.
    """
    print(f"\nEvaluating {model_name} on test set...")

    # Step 1: Get predictions
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)

    # Step 2: Compute metrics
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Step 3: Print results
    print_metrics(metrics, class_names, model_name)

    # Step 4: Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir)

    # Step 5: Classification report
    report = get_classification_report(y_true, y_pred, class_names)
    print("Classification Report:")
    print(report)

    return metrics, y_pred, y_true


def compare_models(metrics_a, metrics_b, model_name_a="Custom CNN", model_name_b="EfficientNetB0",
                   save_dir="outputs/plots"):
    """
    Creates a side-by-side comparison of two models.

    Generates a bar chart comparing accuracy, precision, recall, and F1
    for both models to visually demonstrate which model performs better.

    Args:
        metrics_a (dict): Metrics for Model A (Custom CNN).
        metrics_b (dict): Metrics for Model B (EfficientNet).
        model_name_a (str): Name of Model A.
        model_name_b (str): Name of Model B.
        save_dir (str): Directory to save comparison plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Comparison data
    metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
    values_a = [
        metrics_a['accuracy'],
        metrics_a['precision_macro'],
        metrics_a['recall_macro'],
        metrics_a['f1_macro']
    ]
    values_b = [
        metrics_b['accuracy'],
        metrics_b['precision_macro'],
        metrics_b['recall_macro'],
        metrics_b['f1_macro']
    ]

    # Create grouped bar chart
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, values_a, width, label=model_name_a, color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, values_b, width, label=model_name_b, color='#FF9800', alpha=0.85)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    ax.set_title('Model Comparison: Custom CNN vs EfficientNetB0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to: {save_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  Model Comparison Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {model_name_a:>15} {model_name_b:>15}")
    print(f"  {'-'*50}")
    for name, va, vb in zip(['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
                            values_a, values_b):
        better = "←" if va > vb else "→" if vb > va else "="
        print(f"  {name:<20} {va:>14.2f}% {vb:>14.2f}% {better}")
    print(f"{'='*60}\n")
