"""Configuration management for CIFAR-10 classification.

This module provides dataclasses for type-safe configuration and YAML loading.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""

    train_size: int = 8000
    val_size: int = 2000
    batch_size: int = 32
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    base_model: str = "ResNet50"
    input_shape: tuple = (32, 32, 3)
    num_classes: int = 10
    dropout_rate: float = 0.3
    dense_units: int = 256


@dataclass
class PhaseConfig:
    """Training phase configuration."""

    epochs: int
    learning_rate: float
    freeze_base: bool = True
    unfreeze_from: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration for both phases."""

    phase1: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(epochs=20, learning_rate=0.001, freeze_base=True)
    )
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            epochs=30, learning_rate=0.0001, freeze_base=False, unfreeze_from="conv5_block1"
        )
    )


@dataclass
class CallbackConfig:
    """Callback configuration for training."""

    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    restore_best_weights: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 3
    min_lr: float = 0.00001


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    experiment_name: str = "cifar10-classification"
    tracking_uri: str = "mlruns"


@dataclass
class PathConfig:
    """Path configuration for outputs."""

    models: str = "models"
    outputs: str = "outputs"
    figures: str = "outputs/figures"


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config instance with loaded values.
        """
        path = Path(path)
        with open(path) as f:
            cfg = yaml.safe_load(f)

        data_cfg = DataConfig(**cfg.get("data", {}))

        model_dict = cfg.get("model", {})
        if "input_shape" in model_dict:
            model_dict["input_shape"] = tuple(model_dict["input_shape"])
        model_cfg = ModelConfig(**model_dict)

        training_dict = cfg.get("training", {})
        phase1_dict = training_dict.get("phase1", {})
        phase2_dict = training_dict.get("phase2", {})
        training_cfg = TrainingConfig(
            phase1=PhaseConfig(**phase1_dict),
            phase2=PhaseConfig(**phase2_dict),
        )

        callbacks_dict = cfg.get("callbacks", {})
        callback_cfg = CallbackConfig(
            early_stopping_patience=callbacks_dict.get("early_stopping", {}).get("patience", 5),
            early_stopping_min_delta=callbacks_dict.get("early_stopping", {}).get(
                "min_delta", 0.001
            ),
            restore_best_weights=callbacks_dict.get("early_stopping", {}).get(
                "restore_best_weights", True
            ),
            reduce_lr_factor=callbacks_dict.get("reduce_lr", {}).get("factor", 0.5),
            reduce_lr_patience=callbacks_dict.get("reduce_lr", {}).get("patience", 3),
            min_lr=callbacks_dict.get("reduce_lr", {}).get("min_lr", 0.00001),
        )

        mlflow_cfg = MLflowConfig(**cfg.get("mlflow", {}))
        paths_cfg = PathConfig(**cfg.get("paths", {}))

        return cls(
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            callbacks=callback_cfg,
            mlflow=mlflow_cfg,
            paths=paths_cfg,
        )

    def to_dict(self) -> dict:
        """Convert config to flat dictionary for MLflow logging."""
        return {
            "train_size": self.data.train_size,
            "val_size": self.data.val_size,
            "batch_size": self.data.batch_size,
            "random_seed": self.data.random_seed,
            "base_model": self.model.base_model,
            "input_shape": str(self.model.input_shape),
            "num_classes": self.model.num_classes,
            "dropout_rate": self.model.dropout_rate,
            "dense_units": self.model.dense_units,
            "phase1_epochs": self.training.phase1.epochs,
            "phase1_lr": self.training.phase1.learning_rate,
            "phase2_epochs": self.training.phase2.epochs,
            "phase2_lr": self.training.phase2.learning_rate,
            "unfreeze_from": self.training.phase2.unfreeze_from,
        }
