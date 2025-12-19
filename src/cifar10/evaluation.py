"""Model evaluation utilities.

This module provides functions for evaluating model performance,
computing metrics, and generating visualizations.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from cifar10.data import CLASS_NAMES


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Evaluate model on test set.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: Test labels.

    Returns:
        Tuple of (test_loss, test_accuracy, predictions).
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy, y_pred


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> dict:
    """Generate classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.

    Returns:
        Classification report as dictionary.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix.
        class_names: Optional list of class names.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def get_top_confusions(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    top_n: int = 5,
) -> list:
    """Get top confused class pairs.

    Args:
        cm: Confusion matrix.
        class_names: Optional list of class names.
        top_n: Number of top confusions to return.

    Returns:
        List of tuples (true_class, pred_class, count).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    confusions = []
    for _ in range(top_n):
        idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        true_idx, pred_idx = idx
        count = cm_copy[true_idx, pred_idx]
        if count == 0:
            break
        confusions.append((class_names[true_idx], class_names[pred_idx], int(count)))
        cm_copy[true_idx, pred_idx] = 0

    return confusions


def plot_training_curves(
    history: dict,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training curves (loss and accuracy).

    Args:
        history: Training history dictionary.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    return fig


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot per-class accuracy bar chart.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    report = get_classification_report(y_true, y_pred, class_names)

    accuracies = []
    for name in class_names:
        if name in report:
            accuracies.append(report[name]["precision"])
        else:
            accuracies.append(0.0)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if acc >= 0.5 else "red" for acc in accuracies]
    bars = ax.bar(class_names, accuracies, color=colors, alpha=0.7)

    ax.axhline(y=0.5, color="gray", linestyle="--", label="50% threshold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-class accuracy to {save_path}")

    return fig
