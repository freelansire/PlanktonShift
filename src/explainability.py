from __future__ import annotations
import numpy as np
import tensorflow as tf

def find_last_conv_layer(model: tf.keras.Model) -> str:
    # pick the last Conv2D layer name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise RuntimeError("No Conv2D layer found for Grad-CAM")

def grad_cam_heatmap(model: tf.keras.Model, img: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """
    img: (H,W,3) float32 in [0,1]
    returns heatmap (H,W) in [0,1]
    """
    last_conv = find_last_conv_layer(model)
    conv_layer = model.get_layer(last_conv)

    grad_model = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    x = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heat = tf.reduce_sum(conv_out * pooled, axis=-1)

    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    heat = tf.image.resize(heat[..., None], (img.shape[0], img.shape[1]))[..., 0]
    return heat.numpy()
