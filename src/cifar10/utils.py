"""Utility functions for CIFAR-10 classification.

This module provides common utilities for reproducibility, logging, and path management.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and TensorFlow.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    print(f"Random seeds set to {seed}")


def get_project_root() -> Path:
    """Get the project root directory.

    Searches for pyproject.toml or .git to identify project root.

    Returns:
        Path to project root.
    """
    current = Path.cwd()

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    return current


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level.
        log_file: Optional path to log file.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger("cifar10")
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_gpu_info() -> dict:
    """Get GPU information.

    Returns:
        Dictionary with GPU details.
    """
    gpus = tf.config.list_physical_devices("GPU")

    info = {
        "num_gpus": len(gpus),
        "gpu_names": [],
        "memory_limit": [],
    }

    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info["gpu_names"].append(details.get("device_name", "Unknown"))
        except Exception:
            info["gpu_names"].append(gpu.name)

    return info


def get_environment_info() -> dict:
    """Get environment information for reproducibility logging.

    Returns:
        Dictionary with environment details.
    """
    return {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "gpu_info": get_gpu_info(),
    }


def clear_memory() -> None:
    """Clear GPU memory and garbage collect."""
    import gc

    tf.keras.backend.clear_session()
    gc.collect()

    print("Memory cleared")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path.

    Returns:
        The path (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
