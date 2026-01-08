"""Computer Vision Toolkit.

Classification, detection, interpretability and annotation.
"""

__version__ = "0.2.0"

from cv_toolkit.config import Config
from cv_toolkit.data import load_cifar10, create_dataset, CLASS_NAMES
from cv_toolkit.model import create_base_model, create_classifier_model, unfreeze_layers
from cv_toolkit.training import get_callbacks, compile_model, train_model, train_with_mlflow
from cv_toolkit.evaluation import evaluate_model, plot_confusion_matrix, get_classification_report
from cv_toolkit.gradcam import make_gradcam_heatmap, overlay_heatmap

__all__ = [
    "Config",
    "load_cifar10",
    "create_dataset",
    "CLASS_NAMES",
    "create_base_model",
    "create_classifier_model",
    "unfreeze_layers",
    "get_callbacks",
    "compile_model",
    "train_model",
    "train_with_mlflow",
    "evaluate_model",
    "plot_confusion_matrix",
    "get_classification_report",
    "make_gradcam_heatmap",
    "overlay_heatmap",
]
