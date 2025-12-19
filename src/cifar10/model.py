"""Model architecture for CIFAR-10 classification.

This module provides functions to create the ResNet50 transfer learning model
with a custom classification head.
"""

import tensorflow as tf

from cifar10.config import ModelConfig
from cifar10.data import create_augmentation_layer


def create_base_model(config: ModelConfig) -> tf.keras.Model:
    """Create ResNet50 base model with ImageNet weights.

    Args:
        config: Model configuration.

    Returns:
        ResNet50 base model with frozen weights.
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.input_shape,
        pooling="avg",
    )
    base_model.trainable = False
    return base_model


def create_classifier_model(config: ModelConfig) -> tf.keras.Model:
    """Create full classifier model with ResNet50 base and custom head.

    The architecture is:
        Input -> Augmentation -> ResNet50 Preprocessing -> ResNet50 Base
        -> GlobalAveragePooling -> Dropout -> Dense -> Dropout -> Output

    Args:
        config: Model configuration with dropout_rate, dense_units, num_classes.

    Returns:
        Compiled Keras model ready for training.
    """
    inputs = tf.keras.Input(shape=config.input_shape, name="input")

    x = create_augmentation_layer()(inputs)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = create_base_model(config)
    x = base_model(x, training=False)

    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout1")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_classifier")

    return model


def unfreeze_layers(model: tf.keras.Model, from_layer: str) -> None:
    """Unfreeze layers in the base model from a specific layer onwards.

    Args:
        model: The full classifier model.
        from_layer: Name of the layer to start unfreezing from (e.g., "conv5_block1").
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find base model in classifier")

    base_model.trainable = True

    found = False
    for layer in base_model.layers:
        if from_layer in layer.name:
            found = True
        layer.trainable = found

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    total_count = len(base_model.layers)
    print(f"Unfroze {trainable_count}/{total_count} layers from '{from_layer}'")


def get_model_summary(model: tf.keras.Model) -> str:
    """Get model summary as string.

    Args:
        model: Keras model.

    Returns:
        String representation of model summary.
    """
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def count_parameters(model: tf.keras.Model) -> tuple:
    """Count trainable and non-trainable parameters.

    Args:
        model: Keras model.

    Returns:
        Tuple of (trainable_params, non_trainable_params).
    """
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    return trainable, non_trainable
