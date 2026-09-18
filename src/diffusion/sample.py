"""
diffusion/sample.py — Convenience wrappers for the Streamlit app
======================================================================
"""
import os
import numpy as np
import tensorflow as tf

from src.diffusion.latent_diffusion import (
    LatentDiffusionModel, CLASS_ORDER, IMG_SIZE,
)


def load_for_inference(ckpt_dir: str):
    """Return a loaded LatentDiffusionModel, or None if no checkpoint."""
    if not os.path.exists(os.path.join(ckpt_dir, "unet.keras")):
        return None
    return LatentDiffusionModel.load(ckpt_dir)


def generate_samples(model: LatentDiffusionModel, n: int = 8,
                     classes=None, steps: int = 50) -> np.ndarray:
    """
    Generate n wafer maps.
      classes=None            -> random classes
      classes=list[str]       -> per-sample condition labels
    Returns (n, 52, 52, 1) float32 in [0, 1].
    """
    if classes is None:
        idxs = np.random.randint(0, len(CLASS_ORDER), size=n)
    else:
        idxs = [CLASS_ORDER.index(c) for c in classes]

    outs = []
    for ci in idxs:
        img = model.sample(1, class_idx=int(ci), steps=steps)[0]
        outs.append(img)
    return np.stack(outs).astype(np.float32)
