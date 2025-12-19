"""CIFAR-10 Classification Package.

A production-ready image classification system using ResNet50 transfer learning
with MLflow experiment tracking.

Example:
    from cifar10.config import Config
    from cifar10.data import load_cifar10, create_datasets
    from cifar10.model import create_classifier_model
    from cifar10.training import train_with_mlflow

    config = Config.from_yaml("config/config.yaml")
    x_train, y_train, x_test, y_test = load_cifar10(config.data)
    model = create_classifier_model(config.model)
    history = train_with_mlflow(model, train_ds, val_ds, config, "phase1")
"""

__version__ = "0.2.0"

from cifar10.config import Config, DataConfig, ModelConfig, TrainingConfig
from cifar10.data import load_cifar10, create_datasets, create_augmentation_layer
from cifar10.model import create_classifier_model, unfreeze_layers
from cifar10.training import train_with_mlflow, get_callbacks
from cifar10.evaluation import evaluate_model, plot_confusion_matrix
from cifar10.gradcam import make_gradcam_heatmap, visualize_gradcam
from cifar10.utils import set_seeds, get_project_root

__all__ = [
    # Config
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    # Data
    "load_cifar10",
    "create_datasets",
    "create_augmentation_layer",
    # Model
    "create_classifier_model",
    "unfreeze_layers",
    # Training
    "train_with_mlflow",
    "get_callbacks",
    # Evaluation
    "evaluate_model",
    "plot_confusion_matrix",
    # Grad-CAM
    "make_gradcam_heatmap",
    "visualize_gradcam",
    # Utils
    "set_seeds",
    "get_project_root",
]
