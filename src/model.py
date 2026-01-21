from __future__ import annotations
from typing import Tuple
import tensorflow as tf

def build_model(num_classes: int, image_size: int = 224) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Returns:
      - full model: inputs -> logits
      - feature model: inputs -> embedding (for CORAL / analysis)
    """
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    x = base(inp, training=False)

    x = tf.keras.layers.Dense(256, activation="relu", name="emb")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inp, logits, name="planktonshift")
    feature_model = tf.keras.Model(inp, model.get_layer("emb").output, name="planktonshift_features")
    return model, feature_model

def set_backbone_trainable(model: tf.keras.Model, trainable: bool, fine_tune_at: int = 0) -> None:
    # backbone is MobileNetV3Small nested inside the model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "MobileNetV3" in layer.name:
            backbone = layer
            break
    if backbone is None:
        # fallback: find by name
        for layer in model.layers:
            if "MobileNetV3" in layer.name and hasattr(layer, "layers"):
                backbone = layer
                break

    if backbone is None:
        return

    if not trainable:
        backbone.trainable = False
        return

    backbone.trainable = True
    if fine_tune_at > 0:
        for l in backbone.layers[:fine_tune_at]:
            l.trainable = False
