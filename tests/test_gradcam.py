"""Tests for Grad-CAM module."""

import numpy as np
import tensorflow as tf

from cv_toolkit.gradcam import make_gradcam_heatmap, overlay_heatmap


def test_make_gradcam_heatmap_shape():
    """Test heatmap has correct shape."""
    # Create minimal model for testing
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv_test")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Test image
    img = np.random.rand(32, 32, 3).astype(np.float32)
    
    heatmap = make_gradcam_heatmap(model, img, "conv_test")
    
    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_heatmap_shape():
    """Test overlay produces valid image."""
    img = np.random.rand(32, 32, 3).astype(np.float32)
    heatmap = np.random.rand(32, 32).astype(np.float32)
    
    result = overlay_heatmap(img, heatmap, alpha=0.4)
    
    assert result.shape == (32, 32, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
