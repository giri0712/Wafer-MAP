"""
data_loader.py
Loads and preprocesses the WM-811K wafer map dataset from LSWMD.pkl
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE     = 64
DATA_PATH    = "data/LSWMD.pkl"
OUTPUT_DIR   = "outputs"

DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{path}'.\n"
            "Download WM-811K from:\n"
            "  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "Place LSWMD.pkl in the 'data/' folder."
        )
    print(f"[INFO] Loading dataset from {path} ...")
    import sys
    import pickle
    try:
        import pandas.core.indexes
        sys.modules['pandas.indexes'] = sys.modules['pandas.core.indexes']
    except Exception:
        pass
    with open(path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    print(f"[INFO] Raw shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with a valid failureType label
    df = df.dropna(subset=["failureType"])
    df = df[df["failureType"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)]
    
    # Flatten nested failureType if needed
    def extract_label(x):
        while isinstance(x, (list, np.ndarray)):
            if len(x) > 0:
                x = x[0]
            else:
                return None
        return x

    df = df.copy()
    df["label"] = df["failureType"].apply(extract_label)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(DEFECT_CLASSES)]
    print(f"[INFO] Clean shape after filtering: {df.shape}")
    print(f"\n[INFO] Class distribution:\n{df['label'].value_counts()}\n")
    return df


def preprocess_wafer_map(wafer_map: np.ndarray) -> np.ndarray:
    """Resize and normalize a single wafer map to (IMG_SIZE, IMG_SIZE, 1)."""
    wm = wafer_map.astype(np.float32)
    # Resize
    wm_resized = cv2.resize(wm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    # Normalize to [0, 1]
    if wm_resized.max() > 0:
        wm_resized = wm_resized / wm_resized.max()
    # Add channel dim
    return wm_resized[..., np.newaxis]


def build_arrays(df: pd.DataFrame):
    """Convert df waferMap column → X array, label column → y array."""
    print("[INFO] Preprocessing wafer maps ...")
    X, y = [], []
    for _, row in df.iterrows():
        try:
            wm = np.array(row["waferMap"])
            X.append(preprocess_wafer_map(wm))
            y.append(row["label"])
        except Exception:
            continue

    X = np.array(X, dtype=np.float32)      # (N, 64, 64, 1)
    y = np.array(y)

    # Encode labels
    le = LabelEncoder()
    le.fit(DEFECT_CLASSES)
    y_encoded = le.transform(y)
    y_cat = to_categorical(y_encoded, num_classes=len(DEFECT_CLASSES))

    print(f"[INFO] X shape: {X.shape}, y shape: {y_cat.shape}")
    return X, y_cat, le


def split_data(X, y, random_state=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y.argmax(axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=random_state, stratify=y_tmp.argmax(axis=1)
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = df["label"].value_counts()
    plt.figure(figsize=(12, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("WM-811K Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Defect Pattern")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


def plot_sample_wafer_maps(df: pd.DataFrame, samples_per_class: int = 3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = df["label"].unique()
    fig, axes = plt.subplots(len(classes), samples_per_class,
                             figsize=(samples_per_class * 3, len(classes) * 3))
    fig.suptitle("Sample Wafer Maps per Class", fontsize=14, fontweight="bold", y=1.01)

    for row_idx, cls in enumerate(sorted(classes)):
        samples = df[df["label"] == cls]["waferMap"].head(samples_per_class)
        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx]
            if col_idx < len(samples):
                wm = np.array(samples.iloc[col_idx])
                ax.imshow(wm, cmap="RdYlGn", interpolation="nearest")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sample_wafer_maps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw   = load_dataset()
    df_clean = clean_dataset(df_raw)
    plot_class_distribution(df_clean)
    plot_sample_wafer_maps(df_clean)
    X, y, le = build_arrays(df_clean)
    splits   = split_data(X, y)
    print("[DONE] Data pipeline complete.")
