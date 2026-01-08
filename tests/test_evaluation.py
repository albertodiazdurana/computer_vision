"""Tests for evaluation module."""

import numpy as np
import matplotlib.pyplot as plt

from cv_toolkit.evaluation import plot_confusion_matrix, get_classification_report


def test_plot_confusion_matrix():
    """Test confusion matrix generation."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    
    fig, cm = plot_confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2  # class 0: 2 correct
    assert cm[2, 1] == 1  # class 2 predicted as 1: 1 error
    
    plt.close(fig)


def test_get_classification_report():
    """Test classification report generation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])  # 1 error: predicted 0, was 1
    
    report = get_classification_report(y_true, y_pred)
    
    assert "airplane" in report  # CLASS_NAMES[0]
    assert report["airplane"]["recall"] == 1.0  # found all actual 0s
    assert report["accuracy"] == 0.75  # 3/4 correct

