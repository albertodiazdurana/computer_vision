"""Data loading and preprocessing for CIFAR-10."""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cv_toolkit.config import DataConfig

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def load_cifar10(config: DataConfig):
    """Load CIFAR-10 with train/val/test splits."""
    # Load raw data
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    # Subset to desired size
    total_size = config.train_size + config.val_size
    x_subset, _, y_subset, _ = train_test_split(
        x_train_full, y_train_full,
        train_size=total_size,
        stratify=y_train_full,
        random_state=config.random_seed,
    )

    # Split into train/val
    val_ratio = config.val_size / total_size
    x_train, x_val, y_train, y_val = train_test_split(
        x_subset, y_subset,
        test_size=val_ratio,
        stratify=y_subset,
        random_state=config.random_seed,
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_dataset(x, y, batch_size, shuffle=True):
    """Create tf.data.Dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
