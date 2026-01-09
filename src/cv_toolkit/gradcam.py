"""Grad-CAM implementation for model interpretability."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def make_gradcam_heatmap(model, img, layer_name):
    """Generate Grad-CAM heatmap for an image.
    
    Args:
        model: Trained Keras model
        img: Input image array (H, W, 3)
        layer_name: Name of convolutional layer to visualize
        
    Returns:
        Heatmap array (H, W) normalized to [0, 1]
    """
    # Create model that outputs conv layer activations and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        pred_class = tf.argmax(predictions[0])
        class_output = predictions[:, pred_class]
    
    # Gradient of predicted class w.r.t. conv layer output
    grads = tape.gradient(class_output, conv_outputs)
    
    # Average gradient across spatial dimensions (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalize
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    # Resize to input image size
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (img.shape[0], img.shape[1])
    )
    heatmap = tf.squeeze(heatmap).numpy()
    
    return heatmap



def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image.
    
    Args:
        img: Original image array (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Blending factor for heatmap
        
    Returns:
        Blended image array (H, W, 3) normalized to [0, 1]
    """
    import matplotlib.cm as cm
    
    # Apply jet colormap to heatmap
    colormap = plt.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3]  # Drop alpha channel
    
    # Ensure img is in [0, 1] range
    img_normalized = img.astype(np.float32)
    if img_normalized.max() > 1.0:
        img_normalized = img_normalized / 255.0
    
    # Alpha blend
    blended = (1 - alpha) * img_normalized + alpha * heatmap_rgb
    blended = np.clip(blended, 0.0, 1.0)
    
    return blended.astype(np.float32)

