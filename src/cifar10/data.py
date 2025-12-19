"""Data loading and preprocessing for CIFAR-10.

This module handles loading the CIFAR-10 dataset, creating train/val/test splits,
and applying data augmentation.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cifar10.config import DataConfig

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10(
    config: DataConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset with train/val/test splits.

    Args:
        config: Data configuration with train_size, val_size, and random_seed.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    total_size = config.train_size + config.val_size
    if total_size < len(x_train_full):
        x_subset, _, y_subset, _ = train_test_split(
            x_train_full,
            y_train_full,
            train_size=total_size,
            stratify=y_train_full,
            random_state=config.random_seed,
        )
    else:
        x_subset, y_subset = x_train_full, y_train_full

    val_ratio = config.val_size / (config.train_size + config.val_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_subset,
        y_subset,
        test_size=val_ratio,
        stratify=y_subset,
        random_state=config.random_seed,
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_augmentation_layer() -> tf.keras.Sequential:
    """Create data augmentation layer.

    Returns:
        Keras Sequential model with augmentation layers.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def create_datasets(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create tf.data.Dataset from numpy arrays.

    Args:
        x: Image data array.
        y: Label array.
        batch_size: Batch size for training.
        augment: Whether to apply data augmentation.
        shuffle: Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset ready for training/evaluation.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    if augment:
        augmentation = create_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_class_names() -> list:
    """Get CIFAR-10 class names.

    Returns:
        List of 10 class names.
    """
    return CLASS_NAMES.copy()
