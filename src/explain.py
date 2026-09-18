"""
explain.py — Unified explainability for both architectures
======================================================================
- CNN  -> Grad-CAM on the last conv block (src/gradcam.py)
- ViT  -> attention rollout over all transformer blocks

attention rollout (Abnar & Zuidema, 2020): recursively multiply the
attention matrices A_i with identity + residual, normalized per row:
    R_{i+1} = A_{i+1} (I + R_i)   -> averaged over heads.
The result attributes importance to *input patches*, which we reshape
to the patch grid and upscale to image size for overlay.
"""

import numpy as np
import cv2
import tensorflow as tf

from src.gradcam import make_gradcam_heatmap, overlay_gradcam  # noqa: F401


# ----------------------------------------------------------------------
# ViT attention rollout
# ----------------------------------------------------------------------
def attention_rollout_map(model: tf.keras.Model, img_array: np.ndarray,
                          variant_key: str = "vit") -> np.ndarray:
    """
    Args:
        model: built by src/models/vit.py (contains block_{i}_mha layers)
        img_array: (1, H, W, 1) preprocessed input
    Returns:
        (H, W) rollout heatmap in [0, 1]
    """
    img = tf.cast(img_array, tf.float32)

    # 1) patch embedding pass
    patch_layer = model.get_layer("patch_extract")
    tokens = patch_layer(img, training=False)          # (1, N, D)
    n_tokens = int(tokens.shape[1])
    grid = int(np.sqrt(n_tokens))
    if grid * grid != n_tokens:
        raise ValueError(f"Token count {n_tokens} is not a square grid")

    # 2) walk the transformer blocks, capturing attention weights
    rollout = tf.eye(n_tokens, batch_shape=[1])        # (1, N, N)
    for i in range(50):                                # DEPTH upper bound
        name = f"block_{i}_mha"
        try:
            mha = model.get_layer(name)
        except ValueError:
            break
        ln1 = model.get_layer(f"block_{i}_ln1")
        # recompute the block input = output of previous residual stream
        # (we track it explicitly through the graph below)
        attention_out, scores = mha(tokens, tokens,
                                    return_attention_scores=True)
        tokens = ln1(tokens + attention_out)

        # MLP half of the block (fc1 -> fc2 with residual + LN)
        try:
            fc1 = model.get_layer(f"block_{i}_fc1")
            fc2 = model.get_layer(f"block_{i}_fc2")
            ln2 = model.get_layer(f"block_{i}_ln2")
            tokens = ln2(tokens + fc2(fc1(tokens)))
        except ValueError:
            pass

        scores = tf.reduce_mean(scores, axis=1)         # avg heads (1, N, N)
        eye = tf.eye(n_tokens, batch_shape=[1])
        scores = (scores + eye) / 2.0                   # residual connection
        rollout = tf.matmul(scores, rollout)

    # 3) importance per input patch = mean attention received
    heat = tf.reduce_mean(rollout, axis=-1)[0]          # (N,)
    heat = heat.numpy().reshape(grid, grid).astype(np.float32)
    heat = (heat - heat.min()) / (np.ptp(heat) + 1e-8)

    # 4) upscale to image grid
    h, w = img_array.shape[1], img_array.shape[2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(heat, 0, 1)


def explain(model: tf.keras.Model, img_array: np.ndarray,
            variant_key: str, layer: str = None):
    """Return (heatmap, overlay_bgr) for either architecture."""
    if variant_key == "cnn":
        hm = make_gradcam_heatmap(img_array, model,
                                  last_conv_layer_name=layer or "conv5_b")
        ov = overlay_gradcam(img_array[0], hm)
    else:
        hm = attention_rollout_map(model, img_array, variant_key)
        ov = overlay_gradcam(img_array[0], hm)
    return hm, ov


def explain_grid_figure(model, images, labels, variant_key, layer=None,
                        n_show=8, cmap="magma"):
    """(2, n) grid: row 1 = input maps, row 2 = explanation overlays."""
    import matplotlib.pyplot as plt
    n = min(n_show, len(images))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 4.6))
    for i in range(n):
        x = images[i:i + 1]
        hm, ov = explain(model, x, variant_key, layer)
        axes[0, i].imshow(x[0].squeeze(), cmap="RdYlGn")
        axes[0, i].set_title(labels[i], fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[1, 0].set_ylabel("explanation", fontsize=8)
    plt.tight_layout()
    return fig
