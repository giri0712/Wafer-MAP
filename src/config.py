"""
config.py
======================================================================
Shared configuration for:
  "Diffusion-based minority-class augmentation + lightweight ViT
   classifier with Grad-CAM explainability for mixed-type wafer
   defect detection — a cross-dataset robustness study
   (WM-811K -> MixedWM38)"
"""

import os

# ----------------------------------------------------------------------
# Core dims / classes
# ----------------------------------------------------------------------
IMG_SIZE    = 52            # native MixedWM38 resolution; WM-811K is resized
NUM_CLASSES = 9

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]

# Classes that are minority / under-represented in WM-811K
MINORITY_CLASSES = ["Donut", "Edge-Loc", "Loc", "Near-full", "Scratch", "Random"]

# ----------------------------------------------------------------------
# Per-class display metadata
# ----------------------------------------------------------------------
CLASS_COLORS = {
    "Center":    "#ef4444",
    "Donut":     "#f97316",
    "Edge-Loc":  "#eab308",
    "Edge-Ring": "#f59e0b",
    "Loc":       "#22c55e",
    "Near-full": "#6b7280",
    "Random":    "#3b82f6",
    "Scratch":   "#a855f7",
    "none":      "#2dd4bf",
}

CLASS_ICONS = {
    "Center":    "🔴",
    "Donut":     "🟠",
    "Edge-Loc":  "🟡",
    "Edge-Ring": "🔶",
    "Loc":       "🟢",
    "Near-full": "⚫",
    "Random":    "🔵",
    "Scratch":   "🟣",
    "none":      "✅",
}

CLASS_INFO = {
    "Center":    "Defects clustered at the wafer center — chuck contamination, spin-coat anomalies.",
    "Donut":     "Ring-shaped defect band inside the edge ring — focus / exposure non-uniformity.",
    "Edge-Loc":  "Localized defects at the wafer edge — edge-bead-removal issues.",
    "Edge-Ring": "Full peripheral ring of defects — EBR or CMP edge problems.",
    "Loc":       "Isolated local cluster — particle contamination or mask defect.",
    "Near-full": "Nearly all dies failing — catastrophic process failure.",
    "Random":    "No spatial pattern — random particle events or process noise.",
    "Scratch":   "Linear trail of defects — handling, chuck, or probe damage.",
    "none":      "No defect pattern — clean wafer, high yield.",
}

SEVERITY = {
    "Near-full": "critical",
    "Edge-Ring": "critical",
    "Scratch":   "warning",
    "Donut":     "warning",
    "Center":    "warning",
    "Edge-Loc":  "warning",
    "Loc":       "info",
    "Random":    "info",
    "none":      "info",
}

SEVERITY_COLORS = {
    "critical": "#ef4444",
    "warning":  "#f59e0b",
    "info":     "#3b82f6",
}

# ----------------------------------------------------------------------
# Datasets (kagglehub handles download on Colab)
# ----------------------------------------------------------------------
DATASETS = {
    "wm811k": {
        "name": "WM-811K",
        "kagglehub_id": "qingyi/wm811k-wafer-map",
        "local_path": "data/LSWMD.pkl",
        "n_maps": 811_457,
    },
    "mixedwm38": {
        "name": "MixedWM38",
        "kagglehub_id": "co1d7era/mixedtype-wafer-defect-datasets",
        "local_path": "data/MixedWM38",
        "n_maps": 38_015,
    },
    "sample_wafermap": {
        "name": "Sample-WaferMap",
        "kagglehub_id": "emphymachine/sample-wafermap-data",
        "local_path": "data/sample_wafermap",
        "n_maps": None,
    },
}

# ----------------------------------------------------------------------
# Model variants (3 experiment arms)
# ----------------------------------------------------------------------
OUT_ROOT = "outputs"
DIFFUSION_CKPT_DIR = os.path.join(OUT_ROOT, "diffusion")
CLASSIFIER_ROOT   = os.path.join(OUT_ROOT, "classifier")

MODEL_VARIANTS = {
    "cnn": {
        "key": "cnn",
        "name": "CNN Baseline",
        "desc": "6-block CNN (Wu et al.-style), ~310K params. Ablation baseline.",
        "path": os.path.join(CLASSIFIER_ROOT, "cnn", "best_model.keras"),
        "explain": "gradcam",
        "layer": "conv5_b",
        "color": "#94a3b8",
    },
    "vit": {
        "key": "vit",
        "name": "Lightweight ViT",
        "desc": "ViT-Tiny for wafer maps: 8x8 patches, 6 encoder blocks, ~400K params.",
        "path": os.path.join(CLASSIFIER_ROOT, "vit", "best_model.keras"),
        "explain": "attention",
        "layer": "block_5",   # last transformer encoder block
        "color": "#22d3ee",
    },
    "vit_diff": {
        "key": "vit_diff",
        "name": "ViT + Diffusion Aug",
        "desc": "Same ViT trained with latent-diffusion minority-class augmentation.",
        "path": os.path.join(CLASSIFIER_ROOT, "vit_diff", "best_model.keras"),
        "explain": "attention",
        "layer": "block_5",
        "color": "#a78bfa",
    },
}

DEFAULT_VARIANT = "vit"


def available_variants() -> dict:
    """Return only the model variants whose checkpoints exist on disk."""
    out = {}
    for key, meta in MODEL_VARIANTS.items():
        if os.path.exists(meta["path"]):
            out[key] = meta
    if not out:
        out[DEFAULT_VARIANT] = MODEL_VARIANTS[DEFAULT_VARIANT]
    return out
