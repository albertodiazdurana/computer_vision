"""Evaluation metrics and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from cv_toolkit.data import CLASS_NAMES

def evaluate_model(model, x_test, y_test):
    """Evaluate model and return predictions."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return y_pred, test_loss, test_accuracy

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    
    return fig, cm

def get_classification_report(y_true, y_pred, class_names=None):
    """Generate classification report as dictionary."""
    if class_names is None:
        # Use only the classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [CLASS_NAMES[i] for i in unique_classes]
    
    return classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True
    )

