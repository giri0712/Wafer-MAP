"""
data/multi.py — Unified multi-dataset loader
======================================================================
Loads WM-811K / MixedWM38 / sample-wafermap into a shared format:

    X: (N, 52, 52, 1) float32 in [0, 1]
       die encoding: 1.0 = fail die, 0.0 = pass die, 0.5 = outside-wafer
       (the "none"/background channel semantic — same convention for
        every source so classifiers can't cheat via dataset fingerprints)
    y: int32 labels in the shared 9-class space (see CLASS_NAMES)
    dataset_id: str per source

MixedWM38 ships as .mat files (MATLAB) with variables waferMap + failureType
(or as precomputed .npy stacks on some mirrors); we handle both.
The sample-wafermap dataset (kagglehub emphymachine/sample-wafermap-data)
ships .pkl/.npz waferMap arrays; handled generically.
"""

import os
import glob
import pickle
import numpy as np
import cv2

from src.config import CLASS_NAMES, IMG_SIZE

# ----------------------------------------------------------------------
# Shared label space:  9 classes, index = position in CLASS_NAMES
# ----------------------------------------------------------------------
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# MixedWM38 label strings (from the WaferMap repo / .mat files)
MIXEDWM38_LABELS = {
    "Center": "Center", "Donut": "Donut", "Edge-Loc": "Edge-Loc",
    "Edge-Ring": "Edge-Ring", "Loc": "Loc", "Near-full": "Near-full",
    "Random": "Random", "Scratch": "Scratch", "none": "none",
    # mixed-type aliases present in MixedWM38 (e.g. 'Center-Loc')
    # are kept as their FIRST component for cross-dataset mapping:
}
MIXED_ALIASES = {
    "edge-ring": "Edge-Ring", "edge_loc": "Edge-Loc", "edgeloc": "Edge-Loc",
    "near-full": "Near-full", "nearfull": "Near-full",
    "center-loc": "Center", "center-scratch": "Center",
    "edge-ring-loc": "Edge-Ring", "donut-center": "Donut",
    "scratch-edge-ring": "Edge-Ring", "loc-scratch": "Loc",
    "random-loc": "Loc", "none": "none",
}


def normalize_label(raw) -> str:
    s = str(raw).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    for k, v in MIXED_ALIASES.items():
        if s == k:
            return v
    for name in CLASS_NAMES:
        if s == name.lower():
            return name
    # first-component fallback for mixed types like "center-loc"
    first = s.split("-")[0]
    for name in CLASS_NAMES:
        if first == name.lower():
            return name
    return "none"


def encode_wafer_map(wm: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Raw wafer maps use int codes {0: pass, 1: fail, 2: invalid/none}
    (WM-811K convention). Map to {0.0, 1.0, 0.5}, resize NEAREST, add channel.
    """
    wm = np.asarray(wm)
    if wm.max() > 2:                       # some mirrors use 0/1/2/3 codes
        wm = np.clip(wm, 0, 2)
    out = np.zeros(wm.shape, np.float32)
    out[wm == 1] = 1.0                     # fail die
    out[wm == 2] = 0.5                     # invalid / outside circle
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    return out[..., np.newaxis]


# ----------------------------------------------------------------------
# WM-811K
# ----------------------------------------------------------------------
def load_wm811k(path: str = "data/LSWMD.pkl", with_test: bool = True):
    import sys
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LSWMD.pkl not found at {path} — download from Kaggle "
            "(qingyi/wm811k-wafer-map) or via kagglehub.")
    try:
        import pandas.core.indexes
        sys.modules["pandas.indexes"] = sys.modules["pandas.core.indexes"]
    except Exception:
        pass
    with open(path, "rb") as f:
        df = pickle.load(f, encoding="latin1")

    df = df.dropna(subset=["failureType"])
    df = df[df["failureType"].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)]

    def extract(x):
        while isinstance(x, (list, np.ndarray)):
            x = x[0]
        return x

    labels, maps = [], []
    for _, row in df.iterrows():
        try:
            lab = normalize_label(extract(row["failureType"]))
            maps.append(encode_wafer_map(np.array(row["waferMap"])))
            labels.append(LABEL_TO_IDX[lab])
        except Exception:
            continue
    X = np.stack(maps).astype(np.float32)
    y = np.array(labels, np.int32)
    # WM-811K has train/test columns; honor them when requested
    if with_test and "trianTestLabel" in df.columns:
        def is_train(x):
            try:
                return "Training" in str(extract(x))
            except Exception:
                return True
        mask = df.reset_index(drop=True).apply(
            lambda r: is_train(r.get("trianTestLabel", "Training")), axis=1
        ).values[:len(y)]
        return X[mask], y[mask], X[~mask], y[~mask]
    return X, y, None, None


# ----------------------------------------------------------------------
# MixedWM38 (.mat files)
# ----------------------------------------------------------------------
def load_mixedwm38(root: str = "data/MixedWM38"):
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("pip install scipy — needed for MixedWM38 .mat") from e
    if not os.path.isdir(root):
        root_candidates = glob.glob(os.path.join("data", "**", "MixedWM38*"),
                                    recursive=True)
        if not root_candidates:
            raise FileNotFoundError(
                f"MixedWM38 not found under {root}. Place the kagglehub "
                "download (co1d7era/mixedtype-wafer-defect-datasets) there.")
        root = root_candidates[0]

    maps, labels = [], []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.mat"),
                              recursive=True)):
        try:
            mat = loadmat(f)
            wm = np.array(mat["waferMap"])
            raw = str(np.squeeze(mat["failureType"][0][0]))
            maps.append(encode_wafer_map(wm))
            labels.append(LABEL_TO_IDX[normalize_label(raw)])
        except Exception:
            continue
    if not maps:
        raise FileNotFoundError(f"No .mat files parsed under {root}")
    return np.stack(maps).astype(np.float32), np.array(labels, np.int32)


# ----------------------------------------------------------------------
# sample-wafermap (kagglehub emphymachine/sample-wafermap-data)
# ----------------------------------------------------------------------
def load_sample_wafermap(root: str = "data/sample_wafermap"):
    if not os.path.isdir(root):
        found = glob.glob(os.path.join("data", "**", "*wafermap*"),
                          recursive=True)
        if not found:
            raise FileNotFoundError(
                f"sample-wafermap data not found under {root}")
        root = os.path.dirname(found[0]) if os.path.isfile(found[0]) else found[0]

    maps, labels = [], []
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext in (".npz", ".npy"):
                obj = np.load(f, allow_pickle=True)
                arrs = [obj[k] for k in obj.files] if ext == ".npz" else [obj]
                for arr in arrs:
                    if isinstance(arr, np.ndarray) and arr.ndim in (2, 3):
                        maps.append(encode_wafer_map(arr))
                        labels.append(LABEL_TO_IDX["none"])   # unlabeled source
            elif ext == ".pkl":
                with open(f, "rb") as fh:
                    obj = pickle.load(fh, encoding="latin1")
                rows = obj["waferMap"] if isinstance(obj, dict) else obj
                labs = obj.get("failureType", [None] * len(rows)) \
                    if isinstance(obj, dict) else [None] * len(rows)
                for wm, lab in zip(rows, labs):
                    maps.append(encode_wafer_map(np.array(wm)))
                    labels.append(LABEL_TO_IDX[normalize_label(lab or "none")])
        except Exception:
            continue
    if not maps:
        raise FileNotFoundError(f"No parseable arrays under {root}")
    return np.stack(maps).astype(np.float32), np.array(labels, np.int32)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
LOADERS = {
    "wm811k": load_wm811k,
    "mixedwm38": load_mixedwm38,
    "sample_wafermap": load_sample_wafermap,
}


def load_dataset_arrays(name: str):
    """Return (X, y) for one of the three registered datasets."""
    if name == "wm811k":
        X_tr, y_tr, X_te, y_te = load_wm811k()
        if X_te is not None and len(X_te):
            X = np.concatenate([X_tr, X_te]); y = np.concatenate([y_tr, y_te])
            return X, y
        return X_tr, y_tr
    return LOADERS[name]()


def load_train_arrays(dataset: str = "wm811k", val_frac: float = 0.1,
                      seed: int = 42):
    """
    Train/val split used by every arm (identical split for fairness).
    Saves cache under outputs/cache to avoid re-parsing the 2 GB pickle.
    """
    os.makedirs("outputs/cache", exist_ok=True)
    cache = f"outputs/cache/{dataset}_X.npy"
    cache_y = f"outputs/cache/{dataset}_y.npy"
    if os.path.exists(cache) and os.path.exists(cache_y):
        X = np.load(cache); y = np.load(cache_y)
    else:
        X, y = load_dataset_arrays(dataset)
        np.save(cache, X); np.save(cache_y, y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    return X[idx[n_val:]], y[idx[:n_val]], X[idx[:n_val]], y[idx[n_val:]]
