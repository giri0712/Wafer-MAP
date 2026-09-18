"""
train.py — 3-arm classifier training with ablation export
======================================================================
Arms (identical data split, identical epochs budget for fairness):
  1. cnn       — CNN baseline (class-weighted)
  2. vit       — Lightweight ViT (class-weighted)
  3. vit_diff  — Lightweight ViT + diffusion-synthesized minority classes

Run:
  python src/train.py                 # all arms
  python src/train.py --arms vit vit_diff
  python src/train.py --max-per-class 500   # quick smoke run
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLASS_NAMES, MODEL_VARIANTS, OUT_ROOT, CLASSIFIER_ROOT, MINORITY_CLASSES,
)
from src.models.vit import build_lightweight_vit, compile_model as compile_vit
from src.models.cnn import build_cnn, compile_model as compile_cnn
from src.data import multi

EPOCHS = 40
BATCH = 64
SEED = 42


# ----------------------------------------------------------------------
def class_weights(y_int) -> dict:
    cls = np.unique(y_int)
    w = compute_class_weight("balanced", classes=cls, y=y_int)
    return {int(c): float(x) for c, x in zip(cls, w)}


def callbacks(variant_key: str):
    d = os.path.join(CLASSIFIER_ROOT, variant_key)
    os.makedirs(d, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=8,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(d, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5,
                                             patience=4, min_lr=1e-6),
    ]


def plot_history(history, variant_key: str):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training — {MODEL_VARIANTS[variant_key]['name']}")
    a1.plot(history.history["accuracy"], label="train")
    a1.plot(history.history["val_accuracy"], label="val")
    a1.set_title("Accuracy"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(history.history["loss"], label="train")
    a2.plot(history.history["val_loss"], label="val")
    a2.set_title("Loss"); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, f"training_curves_{variant_key}.png")
    plt.savefig(p, dpi=130); plt.close()
    print(f"[INFO] Saved → {p}")


# ----------------------------------------------------------------------
# Diffusion augmentation: synthesize minority-class maps and append
# ----------------------------------------------------------------------
def diffusion_augment(X_train, y_train, per_class_cap: int = 300):
    """
    Generate synthetic minority-class maps with the trained latent
    diffusion model and append them to the training set.
    (Real data is NEVER replaced — augmentation only.)
    """
    from src.diffusion.latent_diffusion import LatentDiffusionModel
    ckpt = os.path.join(OUT_ROOT, "diffusion")
    if not os.path.exists(os.path.join(ckpt, "unet.keras")):
        print("[WARN] No diffusion checkpoint — skipping augmentation arm. "
              "Run src/train_diffusion.py first.")
        return X_train, y_train

    model = LatentDiffusionModel.load(ckpt)
    synth_X, synth_y = [], []
    for cls in MINORITY_CLASSES:
        ci = CLASS_NAMES.index(cls)
        have = int((y_train == ci).sum())
        need = min(max(per_class_cap - have, 0), per_class_cap)
        if need == 0:
            continue
        print(f"[AUG] {cls}: have {have}, generating {need} …")
        for _ in range(need):
            synth_X.append(model.sample(1, class_idx=ci, steps=50)[0])
            synth_y.append(ci)
    if not synth_X:
        return X_train, y_train
    X_aug = np.concatenate([X_train, np.stack(synth_X).astype(np.float32)])
    y_aug = np.concatenate([y_train, np.array(synth_y, np.int32)])
    perm = np.random.default_rng(SEED).permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ----------------------------------------------------------------------
def run_arm(arm: str, X_tr, y_tr, X_val, y_val, num_classes: int) -> dict:
    tf.keras.utils.set_random_seed(SEED)

    if arm == "cnn":
        model = build_cnn(num_classes)
        compile_cnn(model)
    else:
        model = build_lightweight_vit(num_classes)
        compile_vit(model)

    y_tr_1h = tf.keras.utils.to_categorical(y_tr, num_classes)
    y_val_1h = tf.keras.utils.to_categorical(y_val, num_classes)

    if arm == "vit_diff":
        X_arm, y_arm = diffusion_augment(X_tr, y_tr)
        cw = class_weights(y_arm)      # still weighted — double correction
    else:
        X_arm, y_arm = X_tr, y_tr
        cw = class_weights(y_arm)

    history = model.fit(
        X_arm, y_tr_1h if arm != "vit_diff" else
        tf.keras.utils.to_categorical(y_arm, num_classes),
        validation_data=(X_val, y_val_1h),
        epochs=EPOCHS, batch_size=BATCH,
        class_weight=cw, callbacks=callbacks(arm), verbose=2)

    plot_history(history, arm)
    val_acc = max(history.history["val_accuracy"])
    out = {
        "variant": arm,
        "name": MODEL_VARIANTS[arm]["name"],
        "params": int(model.count_params()),
        "val_acc": float(val_acc),
        "epochs_run": len(history.history["loss"]),
        "train_size": int(len(X_arm)),
    }
    # hold-out predictions for the shared evaluation stage
    preds_path = os.path.join(OUT_ROOT, f"val_preds_{arm}.npy")
    np.save(preds_path, model.predict(X_val, batch_size=128, verbose=0))
    np.save(os.path.join(OUT_ROOT, f"val_y.npy"), y_val)
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["cnn", "vit", "vit_diff"],
                    choices=["cnn", "vit", "vit_diff"])
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap per class for smoke tests")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    X_tr, y_tr, X_val, y_val = multi.load_train_arrays("wm811k")

    if args.max_per_class:                      # smoke-test subsampling
        keep = []
        for c in np.unique(y_tr):
            idx = np.where(y_tr == c)[0][: args.max_per_class]
            keep.extend(idx)
        rng = np.random.default_rng(SEED)
        rng.shuffle(keep)
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    num_classes = len(CLASS_NAMES)
    rows = []
    for arm in args.arms:
        print(f"\n{'='*60}\n[ARM] {arm}\n{'='*60}")
        try:
            rows.append(run_arm(arm, X_tr, y_tr, X_val, y_val, num_classes))
        except Exception as e:
            print(f"[ERROR] arm {arm} failed: {e}")

    # ablation table (merge with any previous rows)
    out_csv = os.path.join(OUT_ROOT, "ablation_results.csv")
    import pandas as pd
    df_new = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        df_new = pd.concat([old[~old.variant.isin(df_new.variant)], df_new])
    df_new.to_csv(out_csv, index=False)
    print(f"\n[DONE] Ablation table → {out_csv}")
    print(df_new.to_string(index=False))


if __name__ == "__main__":
    main()
