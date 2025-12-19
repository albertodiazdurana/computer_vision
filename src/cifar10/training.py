"""Training utilities with MLflow integration.

This module provides training functions with experiment tracking via MLflow.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.keras
import tensorflow as tf

from cifar10.config import CallbackConfig, Config, PhaseConfig


class MLflowCallback(tf.keras.callbacks.Callback):
    """Keras callback to log metrics to MLflow at each epoch."""

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Log metrics at end of each epoch."""
        if logs is None:
            return

        metrics = {
            "train_loss": logs.get("loss"),
            "train_accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(metrics, step=epoch)


def get_callbacks(
    config: CallbackConfig,
    checkpoint_path: Optional[Path] = None,
    include_mlflow: bool = True,
) -> list:
    """Create training callbacks.

    Args:
        config: Callback configuration.
        checkpoint_path: Optional path for ModelCheckpoint.
        include_mlflow: Whether to include MLflow callback.

    Returns:
        List of Keras callbacks.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            restore_best_weights=config.restore_best_weights,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
    ]

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )
        )

    if include_mlflow:
        callbacks.append(MLflowCallback())

    return callbacks


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    """Compile model with optimizer and loss.

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for Adam optimizer.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    phase_config: PhaseConfig,
    callback_config: CallbackConfig,
    checkpoint_path: Optional[Path] = None,
) -> tf.keras.callbacks.History:
    """Train model for one phase.

    Args:
        model: Keras model to train.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        phase_config: Phase training configuration.
        callback_config: Callback configuration.
        checkpoint_path: Optional path for checkpoints.

    Returns:
        Training history.
    """
    compile_model(model, phase_config.learning_rate)

    callbacks = get_callbacks(callback_config, checkpoint_path)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase_config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start_time

    print(f"Training completed in {training_time / 60:.1f} minutes")

    return history


def train_with_mlflow(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Config,
    phase_config: PhaseConfig,
    run_name: str,
    checkpoint_path: Optional[Path] = None,
) -> tf.keras.callbacks.History:
    """Train model with MLflow experiment tracking.

    Args:
        model: Keras model to train.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        config: Full configuration for logging.
        phase_config: Phase training configuration.
        run_name: Name for MLflow run.
        checkpoint_path: Optional path for checkpoints.

    Returns:
        Training history.
    """
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.to_dict())
        mlflow.log_param("run_name", run_name)

        compile_model(model, phase_config.learning_rate)
        callbacks = get_callbacks(config.callbacks, checkpoint_path, include_mlflow=True)

        start_time = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase_config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        training_time = time.time() - start_time

        final_metrics = {
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "final_val_loss": history.history["val_loss"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1],
            "training_time_min": training_time / 60,
            "epochs_completed": len(history.history["loss"]),
        }
        mlflow.log_metrics(final_metrics)

        mlflow.keras.log_model(model, "model")

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        mlflow.log_dict(history_dict, "history.json")

        print(f"Training completed in {training_time / 60:.1f} minutes")
        print(f"Best val accuracy: {final_metrics['best_val_accuracy']:.4f}")

    return history


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    """Save training history to JSON file.

    Args:
        history: Keras training history.
        path: Path to save JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=2)


def load_history(path: Path) -> dict:
    """Load training history from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary with training history.
    """
    with open(path) as f:
        return json.load(f)
