"""
train_diffusion.py — Train the latent-diffusion augmentation model
======================================================================
Trains on the WM-811K **training split only** (no leakage into val/test),
with class-balanced sampling so minority classes get equal UNet exposure.

Run:  python src/train_diffusion.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DIFFUSION_CKPT_DIR, OUT_ROOT, CLASS_NAMES
from src.diffusion.latent_diffusion import LatentDiffusionModel

EPOCHS_VAE = 30
EPOCHS_UNET = 60
BATCH = 64


def balanced_dataset(X, y_int, batch=BATCH, seed=42):
    """Class-balanced sampling so minority classes drive the UNet equally."""
    n_class = len(np.unique(y_int))
    per = max(len(y_int) // n_class, 1)
    idxs = []
    rng = np.random.default_rng(seed)
    for c in np.unique(y_int):
        pool = np.where(y_int == c)[0]
        take = min(per, len(pool))
        idxs.append(rng.choice(pool, size=take, replace=len(pool) < per))
    idxs = rng.permutation(np.concatenate(idxs))
    ds = tf.data.Dataset.from_tensor_slices((X[idxs], y_int[idxs]))
    return ds.shuffle(2048, seed=seed).batch(batch).prefetch(tf.data.AUTOTUNE)


def preview_grid(model, out_path, n_per_class=4):
    classes = CLASS_NAMES
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(n_per_class * 1.6, len(classes) * 1.6))
    for r, cls in enumerate(classes):
        for c in range(n_per_class):
            img = model.sample(1, class_label=cls, steps=50)[0]
            ax = axes[r, c]
            ax.imshow(img.squeeze(), cmap="RdYlGn", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, fontsize=7, rotation=0,
                              labelpad=42, va="center")
    plt.suptitle("Latent-diffusion samples per class", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved preview → {out_path}")


def main():
    from src.data import multi
    X_train, y_train, _, _ = multi.load_train_arrays()   # (N,52,52,1), int labels

    model = LatentDiffusionModel(num_classes=len(CLASS_NAMES))

    # ---- Stage 1: VAE ----
    print("\n[STAGE 1] Training VAE …")
    model.vae.compile(lr=2e-4)
    model.vae.fit(X_train, epochs=EPOCHS_VAE, batch_size=BATCH, verbose=2)

    # ---- Stage 2: class-conditional UNet in latent space ----
    print("\n[STAGE 2] Training class-conditional UNet …")
    model.compile(lr=2e-4)
    y_int = y_train.argmax(1) if y_train.ndim > 1 else y_train
    ds = balanced_dataset(X_train, y_int)
    model.fit(ds, epochs=EPOCHS_UNET, verbose=2)

    model.save(DIFFUSION_CKPT_DIR)
    print(f"[DONE] Diffusion model saved → {DIFFUSION_CKPT_DIR}")

    preview_grid(model, os.path.join(OUT_ROOT, "diffusion_preview.png"))


if __name__ == "__main__":
    main()
