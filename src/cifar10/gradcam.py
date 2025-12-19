"""Grad-CAM implementation for model interpretability.

This module provides Grad-CAM visualization to understand which image regions
the model focuses on for predictions.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cifar10.data import CLASS_NAMES


def get_gradcam_model(
    model: tf.keras.Model,
    layer_name: Optional[str] = None,
) -> Tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """Extract base model and target layer for Grad-CAM.

    Args:
        model: Full classifier model.
        layer_name: Optional specific layer name. If None, uses last conv layer.

    Returns:
        Tuple of (base_model, target_layer).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find base model in classifier")

    if layer_name is None:
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break
        else:
            raise ValueError("No Conv2D layer found in base model")
    else:
        target_layer = base_model.get_layer(layer_name)

    return base_model, target_layer


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for an image.

    Args:
        model: Full classifier model.
        img_array: Single image array with shape (1, H, W, C) or (H, W, C).
        pred_index: Optional class index to visualize. If None, uses predicted class.
        layer_name: Optional specific layer name for Grad-CAM.

    Returns:
        Heatmap array normalized to [0, 1].
    """
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    base_model = None
    base_model_idx = None
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            base_model_idx = idx
            break

    if base_model is None:
        raise ValueError("Could not find base model in classifier")

    if layer_name is None:
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break
        else:
            raise ValueError("No Conv2D layer found")
    else:
        target_layer = base_model.get_layer(layer_name)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            base_model.get_layer(target_layer.name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.

    Args:
        img: Original image array (H, W, C) in [0, 255] or [0, 1].
        heatmap: Grad-CAM heatmap.
        alpha: Overlay transparency.

    Returns:
        Superimposed image as uint8 array.
    """
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (img.shape[0], img.shape[1]),
    ).numpy()[:, :, 0]

    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    superimposed = (heatmap_colored * alpha + img * (1 - alpha)).astype(np.uint8)

    return superimposed


def visualize_gradcam(
    model: tf.keras.Model,
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    n_samples: int = 4,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize Grad-CAM for multiple samples.

    Args:
        model: Trained classifier model.
        images: Image array.
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Optional class names.
        n_samples: Number of samples to visualize.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)

    for i in range(n_samples):
        img = images[i]
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]

        heatmap = make_gradcam_heatmap(model, img, pred_index=y_pred[i])

        superimposed = overlay_heatmap(img, heatmap)

        axes[i, 0].imshow(img.astype(np.uint8) if img.max() > 1 else img)
        axes[i, 0].set_title(f"True: {true_label}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(superimposed)
        correct = "OK" if y_true[i] == y_pred[i] else "X"
        axes[i, 2].set_title(f"Pred: {pred_label} [{correct}]")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM visualization to {save_path}")

    return fig


def visualize_confusion_pairs(
    model: tf.keras.Model,
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confusion_pairs: list,
    class_names: Optional[list] = None,
    samples_per_pair: int = 2,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize Grad-CAM for confused class pairs.

    Args:
        model: Trained classifier model.
        images: Image array.
        y_true: True labels.
        y_pred: Predicted labels.
        confusion_pairs: List of (true_class, pred_class, count) tuples.
        class_names: Optional class names.
        samples_per_pair: Samples to show per confusion pair.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_pairs = len(confusion_pairs)
    fig, axes = plt.subplots(n_pairs, samples_per_pair * 2, figsize=figsize)

    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for pair_idx, (true_class, pred_class, count) in enumerate(confusion_pairs):
        true_idx = class_names.index(true_class)
        pred_idx = class_names.index(pred_class)

        confused_mask = (y_true == true_idx) & (y_pred == pred_idx)
        confused_indices = np.where(confused_mask)[0]

        for sample_idx in range(samples_per_pair):
            if sample_idx < len(confused_indices):
                idx = confused_indices[sample_idx]
                img = images[idx]

                heatmap = make_gradcam_heatmap(model, img, pred_index=pred_idx)
                superimposed = overlay_heatmap(img, heatmap)

                col_original = sample_idx * 2
                col_gradcam = sample_idx * 2 + 1

                axes[pair_idx, col_original].imshow(
                    img.astype(np.uint8) if img.max() > 1 else img
                )
                axes[pair_idx, col_original].set_title(f"True: {true_class}")
                axes[pair_idx, col_original].axis("off")

                axes[pair_idx, col_gradcam].imshow(superimposed)
                axes[pair_idx, col_gradcam].set_title(f"Pred: {pred_class}")
                axes[pair_idx, col_gradcam].axis("off")
            else:
                axes[pair_idx, sample_idx * 2].axis("off")
                axes[pair_idx, sample_idx * 2 + 1].axis("off")

    fig.suptitle("Grad-CAM Analysis: Confused Class Pairs", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion analysis to {save_path}")

    return fig
