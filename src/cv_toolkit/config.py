"""Configuration loading from YAML files."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    train_size: int
    val_size: int
    batch_size: int
    random_seed: int


@dataclass
class ModelConfig:
    base_model: str
    input_shape: tuple
    num_classes: int
    dropout_rate: float
    dense_units: int


@dataclass
class PhaseConfig:
    epochs: int
    learning_rate: float
    unfreeze_from: str = None


@dataclass
class TrainingConfig:
    phase1: PhaseConfig
    phase2: PhaseConfig


@dataclass
class MLflowConfig:
    experiment_name: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MLflowConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            data=DataConfig(**cfg["data"]),
            model=ModelConfig(
                **{**cfg["model"], "input_shape": tuple(cfg["model"]["input_shape"])}
            ),
            training=TrainingConfig(
                phase1=PhaseConfig(**cfg["training"]["phase1"]),
                phase2=PhaseConfig(**cfg["training"]["phase2"]),
            ),
            mlflow=MLflowConfig(**cfg["mlflow"]),
        )
