"""
gradcam.py
Grad-CAM heatmap generation for WaferMapCNN explainability.
Works with both TensorFlow/Keras models.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model,
                          last_conv_layer_name: str = "conv6",
                          pred_index: int = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given input image.

    Args:
        img_array: shape (1, H, W, C) — preprocessed wafer image
        model: trained Keras model
        last_conv_layer_name: name of the last conv layer to hook
        pred_index: class index to explain (None = top prediction)

    Returns:
        heatmap: np.ndarray (H, W) normalized to [0, 1]
    """
    # Build model that outputs last conv activations + final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original wafer map image.

    Args:
        original_img: shape (H, W) or (H, W, 1) — single-channel wafer image
        heatmap: shape (h, w) — grad-cam output (values in [0,1])
        alpha: blend factor for the heatmap overlay
        colormap: OpenCV colormap for heatmap coloring

    Returns:
        overlaid: np.ndarray (H, W, 3) uint8 BGR image
    """
    # Ensure original is (H, W)
    if original_img.ndim == 3:
        original_img = original_img.squeeze()

    H, W = original_img.shape

    # Rescale original to [0, 255]
    orig_uint8 = (original_img * 255).astype(np.uint8)
    orig_bgr   = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2BGR)

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend
    overlaid = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def gradcam_figure(img_array: np.ndarray, model: tf.keras.Model,
                   classes: list, pred_label: str, confidence: float,
                   last_conv_layer: str = "conv6") -> plt.Figure:
    """
    Create a 3-panel Matplotlib figure:
      [Original] | [Grad-CAM Heatmap] | [Overlay]
    """
    heatmap  = make_gradcam_heatmap(img_array, model,
                                     last_conv_layer_name=last_conv_layer)
    original = img_array[0]  # (H, W, 1)
    overlay  = overlay_gradcam(original, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Prediction: {pred_label}  ({confidence:.1%} confidence)",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(original.squeeze(), cmap="RdYlGn", interpolation="nearest")
    axes[0].set_title("Original Wafer Map")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", interpolation="bilinear")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
