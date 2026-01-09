"""Model architecture for CIFAR-10 classification."""

import tensorflow as tf

from cv_toolkit.config import ModelConfig

def create_base_model(config: ModelConfig):
    """Create ResNet50 base model with frozen weights."""
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.input_shape,
        pooling="avg",
    )
    base_model.trainable = False
    return base_model

def create_classifier_model(config: ModelConfig):
    """Create full classifier with ResNet50 base and custom head."""
    # Input layer
    inputs = tf.keras.Input(shape=config.input_shape)
    
    # Preprocessing (scales pixels to [-1, 1] for ResNet)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    # Base model (frozen)
    base_model = create_base_model(config)
    x = base_model(x, training=False)
    
    # Classification head
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs)

def unfreeze_layers(model, from_layer: str):
    """Unfreeze base model layers from a specific layer onwards."""
    # Find the base model (it's a nested model)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        raise ValueError("Could not find base model")
    
    # Unfreeze from the specified layer
    base_model.trainable = True
    found = False
    for layer in base_model.layers:
        if from_layer in layer.name:
            found = True
        layer.trainable = found
    
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze {trainable}/{len(base_model.layers)} layers from '{from_layer}'")
