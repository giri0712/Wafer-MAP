"""
evaluate.py — Cross-dataset robustness benchmark + explainability export
======================================================================
1. Ablation metrics per arm  (accuracy, macro-F1, per-class F1, minority recall)
2. Cross-dataset matrix      (train WM-811K -> test MixedWM38 / sample, + reverse)
3. Explainability grid       (Grad-CAM for CNN, attention rollout for ViT)

Run:
  python src/evaluate.py            # everything that has checkpoints/data
  python src/evaluate.py --skip-explain
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLASS_NAMES, MODEL_VARIANTS, OUT_ROOT, MINORITY_CLASSES,
)
from src.explain import explain, explain_grid_figure
from src.data import multi

MINORITY_IDX = [CLASS_NAMES.index(c) for c in MINORITY_CLASSES]


# ----------------------------------------------------------------------
def load_arm(variant_key: str):
    path = MODEL_VARIANTS[variant_key]["path"]
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


def eval_on(model, X, y):
    prob = model.predict(X, batch_size=128, verbose=0)
    pred = prob.argmax(1)
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro",
                                   labels=list(range(len(CLASS_NAMES))),
                                   zero_division=0)),
        "minority_recall": float(recall_score(
            y, pred, labels=MINORITY_IDX, average="macro",
            zero_division=0)) if len(y) else 0.0,
        "per_class_f1": f1_score(y, pred, average=None,
                                 labels=list(range(len(CLASS_NAMES))),
                                 zero_division=0),
        "y_pred": pred,
    }


def eval_split(X, y, tag, rows, rows_perclass, y_val_ref=None):
    for vk, meta in MODEL_VARIANTS.items():
        model = load_arm(vk)
        if model is None:
            continue
        r = eval_on(model, X, y)
        rows.append({
            "variant": vk, "name": meta["name"], "split": tag,
            "acc": r["acc"], "macro_f1": r["macro_f1"],
            "minority_recall": r["minority_recall"],
            "params": meta and model.count_params(),
        })
        rows_perclass.append({
            "variant": meta["name"],
            **{c: float(f) for c, f in zip(CLASS_NAMES, r["per_class_f1"])},
        })
        # save confusion data for the studio
        cm = confusion_matrix(y, r["y_pred"], labels=list(range(len(CLASS_NAMES))))
        np.save(os.path.join(OUT_ROOT, f"cm_{vk}_{tag}.npy"), cm)
        print(f"  [{meta['name']}] {tag}: acc={r['acc']:.3f} "
              f"macroF1={r['macro_f1']:.3f} minRecall={r['minority_recall']:.3f}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--max-test", type=int, default=4000,
                    help="cap per-dataset test size for speed")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "explain"), exist_ok=True)

    rows, rows_pc = [], []

    # ---------------- in-distribution (WM-811K val) ----------------
    print("\n[1/3] WM-811K validation split …")
    try:
        _, _, X_val, y_val = multi.load_train_arrays("wm811k")
        if args.max_test and len(X_val) > args.max_test:
            X_val, y_val = X_val[: args.max_test], y_val[: args.max_test]
        eval_split(X_val, y_val, "wm811k_val", rows, rows_pc)
    except Exception as e:
        print(f"[WARN] WM-811K eval skipped: {e}")

    # ---------------- cross-dataset: MixedWM38 ----------------
    print("\n[2/3] Cross-dataset: MixedWM38 (zero-shot) …")
    try:
        X38, y38 = multi.load_mixedwm38()
        if args.max_test and len(X38) > args.max_test:
            X38, y38 = X38[: args.max_test], y38[: args.max_test]
        eval_split(X38, y38, "mixedwm38_zeroshot", rows, rows_pc)
    except Exception as e:
        print(f"[WARN] MixedWM38 eval skipped: {e}")

    # ---------------- cross-dataset: sample-wafermap ----------------
    print("\n[3/3] Cross-dataset: sample-wafermap (zero-shot) …")
    try:
        Xs, ys = multi.load_sample_wafermap()
        if len(Xs) > args.max_test:
            Xs, ys = Xs[: args.max_test], ys[: args.max_test]
        eval_split(Xs, ys, "sample_zeroshot", rows, rows_pc)
    except Exception as e:
        print(f"[WARN] sample-wafermap eval skipped: {e}")

    # ---------------- persist ----------------
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_ROOT, "cross_dataset_results.csv"),
                  index=False)
        print(f"\n[INFO] Saved → outputs/cross_dataset_results.csv")
        print(df.to_string(index=False))

    if rows_pc:
        df_pc = pd.DataFrame(rows_pc)
        # keep the in-distribution rows for the per-class chart
        df_pc = df_pc.loc[:, ~df_pc.columns.duplicated()]
        df_pc.to_csv(os.path.join(OUT_ROOT, "per_class_f1.csv"), index=False)
        print(f"[INFO] Saved → outputs/per_class_f1.csv")

    # ---------------- explainability grids ----------------
    if not args.skip_explain:
        print("\n[EXPLAIN] Exporting explanation grids …")
        for vk, meta in MODEL_VARIANTS.items():
            model = load_arm(vk)
            if model is None:
                continue
            try:
                n = min(8, len(X_val))
                sample_X = X_val[:n]
                sample_y = [CLASS_NAMES[i] for i in y_val[:n]]
                fig = explain_grid_figure(model, sample_X, sample_y, vk,
                                          meta.get("layer"))
                p = os.path.join(OUT_ROOT, "explain", f"grid_{vk}.png")
                fig.savefig(p, dpi=130, bbox_inches="tight")
                plt.close(fig)
                print(f"[INFO] Saved → {p}")
            except Exception as e:
                print(f"[WARN] explain for {vk} failed: {e}")

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
