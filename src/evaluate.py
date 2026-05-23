"""
evaluate.py
Full evaluation suite: confusion matrix, ROC curves,
classification report, and top misclassified samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

OUTPUT_DIR  = "outputs"
MODEL_PATH  = "outputs/best_model.keras"


def load_artifacts():
    model   = tf.keras.models.load_model(MODEL_PATH)
    X_test  = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test  = np.load(f"{OUTPUT_DIR}/y_test.npy")
    classes = np.load(f"{OUTPUT_DIR}/label_classes.npy", allow_pickle=True)
    return model, X_test, y_test, classes


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — WaferMapCNN", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Normalized (Recall)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Classification Report ─────────────────────────────────────────────────────
def save_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)
    path = f"{OUTPUT_DIR}/classification_report.txt"
    with open(path, "w") as f:
        f.write("WaferMapCNN — Classification Report\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"[INFO] Saved → {path}")


# ── ROC Curves ────────────────────────────────────────────────────────────────
def plot_roc_curves(y_test_cat, y_prob, classes):
    n_classes = len(classes)
    y_bin = y_test_cat  # already one-hot

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest) — WaferMapCNN", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Top Misclassified ─────────────────────────────────────────────────────────
def plot_top_misclassified(X_test, y_true, y_pred, y_prob, classes, top_n=12):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("[INFO] No misclassified samples found!")
        return

    # Sort by max confidence in wrong prediction (most confidently wrong first)
    wrong_conf = y_prob[wrong_idx].max(axis=1)
    sorted_idx = wrong_idx[np.argsort(wrong_conf)[::-1]][:top_n]

    cols = 4
    rows = (len(sorted_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Top Misclassified Wafer Maps (Most Confident Errors)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i, idx in enumerate(sorted_idx):
        ax = axes[i]
        ax.imshow(X_test[idx].squeeze(), cmap="RdYlGn", interpolation="nearest")
        conf = y_prob[idx].max()
        ax.set_title(
            f"True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]} ({conf:.0%})",
            fontsize=8, color="red"
        )
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/top_misclassified.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading model and test data ...")
    model, X_test, y_test, classes = load_artifacts()

    print("[INFO] Running inference on test set ...")
    y_prob = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    test_acc = (y_pred == y_true).mean()
    print(f"\n[RESULT] Test Accuracy: {test_acc*100:.2f}%\n")

    plot_confusion_matrix(y_true, y_pred, classes)
    save_classification_report(y_true, y_pred, classes)
    plot_roc_curves(y_test, y_prob, classes)
    plot_top_misclassified(X_test, y_true, y_pred, y_prob, classes)

    print(f"\n[DONE] All outputs saved to '{OUTPUT_DIR}/'")
    print("  - confusion_matrix.png")
    print("  - classification_report.txt")
    print("  - roc_curves.png")
    print("  - top_misclassified.png")


if __name__ == "__main__":
    main()
