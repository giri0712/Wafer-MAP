"""
train.py
Trains the WaferMapCNN on WM-811K dataset with early stopping,
learning rate scheduling, and class-imbalance handling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import (
    load_dataset, clean_dataset, build_arrays,
    split_data, plot_class_distribution, plot_sample_wafer_maps
)
from src.model import build_cnn, compile_model

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 32
OUTPUT_DIR  = "outputs"
MODEL_PATH  = "outputs/best_model.keras"
LR          = 0.001


# ── Training ──────────────────────────────────────────────────────────────────
def get_class_weights(y_train: np.ndarray) -> dict:
    y_ints = y_train.argmax(axis=1)
    classes = np.unique(y_ints)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_ints)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[INFO] Class weights: {cw}")
    return cw


def build_callbacks() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]


def train(model, X_train, y_train, X_val, y_val, class_weights):
    callbacks = build_callbacks()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — WaferMapCNN", fontsize=14, fontweight="bold")

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train Acc",  color="#00d4b4", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Acc",    color="#f59e0b", linewidth=2, linestyle="--")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train Loss", color="#00d4b4", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#ef4444", linewidth=2, linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU detected: {gpus}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("[WARN] No GPU detected — training on CPU (will be slower)")

    # Data
    df_raw   = load_dataset()
    df_clean = clean_dataset(df_raw)
    plot_class_distribution(df_clean)
    plot_sample_wafer_maps(df_clean)

    X, y, le = build_arrays(df_clean)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save label encoder classes for inference
    np.save(f"{OUTPUT_DIR}/label_classes.npy", le.classes_)
    print(f"[INFO] Label classes saved → {OUTPUT_DIR}/label_classes.npy")

    # Model
    num_classes = y.shape[1]
    model = build_cnn(num_classes=num_classes)
    model = compile_model(model, learning_rate=LR)
    model.summary()

    # Class weights
    cw = get_class_weights(y_train)

    # Train
    print("\n[INFO] Starting training ...\n")
    history = train(model, X_train, y_train, X_val, y_val, cw)

    # Curves
    plot_training_curves(history)

    # Final test evaluation
    print("\n[INFO] Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"{'='*40}\n")

    # Save test data for evaluate.py
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    print(f"[INFO] Test arrays saved for evaluate.py")
    print("[DONE] Training complete. Run: python src/evaluate.py")


if __name__ == "__main__":
    main()
