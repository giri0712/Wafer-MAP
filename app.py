"""
app.py — WaferDefect Studio
======================================================================
Research console for:
  "Diffusion-based minority-class augmentation + lightweight ViT
   classifier with Grad-CAM explainability for mixed-type wafer
   defect detection — a cross-dataset robustness study
   (WM-811K -> MixedWM38)"

Pages
  1. Predict          — live inference + attention rollout / Grad-CAM
  2. Diffusion Lab    — generate wafer maps from the latent-diffusion model
  3. Experiments      — 3-arm ablation results (CNN / ViT / ViT+Diff)
  4. Cross-Dataset    — robustness matrix WM-811K -> MixedWM38
  5. Explainability   — Grad-CAM vs attention-rollout comparison
  6. About            — methods summary

Run with:  streamlit run app.py
"""

import os
import io
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    IMG_SIZE, CLASS_NAMES, CLASS_COLORS, CLASS_INFO, CLASS_ICONS,
    SEVERITY, SEVERITY_COLORS, MODEL_VARIANTS, DIFFUSION_CKPT_DIR,
    OUT_ROOT, MINORITY_CLASSES, available_variants,
)
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

MINORITY_CLASSES_SET = set(MINORITY_CLASSES)

# ======================================================================
# Theme palettes  (sidebar picker re-rolls the whole UI accent system)
# ======================================================================
PALETTES = {
    "Nebula Dusk": {
        "bg": "#0b0f1c", "panel": "#131a2e", "border": "#232d4d",
        "accents": ["#7c3aed", "#22d3ee", "#f472b6", "#34d399",
                    "#fbbf24", "#60a5fa", "#fb7185", "#a78bfa", "#2dd4bf"],
        "title": "#e2e8f0", "muted": "#94a3b8", "cmap": "magma",
    },
    "Terminal Ice": {
        "bg": "#05080d", "panel": "#0d1420", "border": "#1b2740",
        "accents": ["#38bdf8", "#818cf8", "#22d3ee", "#a78bfa", "#34d399",
                    "#fbbf24", "#f472b6", "#60a5fa", "#2dd4bf"],
        "title": "#e2e8f0", "muted": "#7dd3fc", "cmap": "viridis",
    },
    "Forest Ember": {
        "bg": "#0a0f0d", "panel": "#101816", "border": "#1e2f2a",
        "accents": ["#f59e0b", "#34d399", "#fbbf24", "#a3e635", "#10b981",
                    "#f97316", "#fde047", "#6ee7b7", "#d946ef"],
        "title": "#e7e6e3", "muted": "#9ca3af", "cmap": "inferno",
    },
    "Sakura Dark": {
        "bg": "#120d12", "panel": "#1b1319", "border": "#33222e",
        "accents": ["#f472b6", "#c084fc", "#fbbf24", "#60a5fa", "#34d399",
                    "#fb7185", "#a78bfa", "#f9a8d4", "#2dd4bf"],
        "title": "#fce7f3", "muted": "#d8b4fe", "cmap": "pink_r",
    },
    "Solar Flare": {
        "bg": "#0d0a05", "panel": "#161008", "border": "#332411",
        "accents": ["#f59e0b", "#fb7185", "#fbbf24", "#fbbf24", "#a3e635",
                    "#34d399", "#6ee7b7", "#f97316", "#d946ef"],
        "title": "#f4f1ea", "muted": "#a8a29e", "cmap": "hot",
    },
}
DEFAULT_PALETTE = "Nebula Dusk"

PAGES = ["🔮 Predict", "🌫 Diffusion Lab", "📊 Experiments",
         "🌐 Cross-Dataset", "🔍 Explainability", "ℹ About"]


def hex_to_rgba(h: str, a: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def build_css(p: dict) -> str:
    return f"""
    <style>
        .stApp {{
            background: radial-gradient(1200px 600px at 80% -10%,
                {hex_to_rgba(p['accents'][0], 0.12)}, transparent),
                radial-gradient(900px 500px at -10% 110%,
                {hex_to_rgba(p['accents'][1], 0.10)}, transparent),
                {p['bg']};
            color: {p['title']};
        }}
        section[data-testid="stSidebar"] {{
            background: {p['panel']};
            border-right: 1px solid {p['border']};
        }}
        h1, h2, h3 {{ color: {p['title']} !important; letter-spacing: 0.2px; }}
        .gradient-title {{
            font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;
            background: linear-gradient(90deg, {p['accents'][0]},
                {p['accents'][1]}, {p['accents'][2]});
            -webkit-background-clip: text; background-clip: text;
            color: transparent; margin-bottom: 0;
        }}
        .subtle {{ color: {p['muted']}; font-size: 0.95rem; }}
        div[data-testid="stMetric"] {{
            background: {p['panel']}; border: 1px solid {p['border']};
            border-radius: 14px; padding: 14px 16px;
        }}
        div[data-testid="stMetric"] label {{
            color: {p['muted']} !important;
        }}
        .badge {{
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px;
            border: 1px solid; margin-right: 6px;
        }}
        .card {{
            background: {p['panel']}; border: 1px solid {p['border']};
            border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {p['panel']}; border: 1px solid {p['border']};
            border-radius: 10px 10px 0 0; padding: 8px 18px;
        }}
        .stTabs [aria-selected="true"] {{
            border-color: {p['accents'][0]};
            box-shadow: inset 0 -2px 0 {p['accents'][0]};
        }}
        .stProgress > div > div {{ background: linear-gradient(90deg,
            {p['accents'][0]}, {p['accents'][1]}); }}
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-thumb {{
            background: {p['border']}; border-radius: 8px;
        }}
        div[data-testid="stFileUploaderDropzone"] {{
            border: 1.5px dashed {p['accents'][1]};
            background: {hex_to_rgba(p['accents'][1], 0.05)};
        }}
    </style>
    """


def class_color(cls: str, pal: dict) -> str:
    return CLASS_COLORS.get(cls, pal["accents"][0])


# ======================================================================
# Cached loaders
# ======================================================================
@st.cache_resource(show_spinner="Loading model …")
def load_model(variant_key: str):
    meta = MODEL_VARIANTS[variant_key]
    if not os.path.exists(meta["path"]):
        return None
    return tf.keras.models.load_model(meta["path"], compile=False)


@st.cache_resource(show_spinner="Loading diffusion decoder …")
def load_diffusion():
    try:
        from src.diffusion.sample import load_for_inference
        return load_for_inference(DIFFUSION_CKPT_DIR)
    except Exception:
        return None


@st.cache_data
def load_csv_safe(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def variant_options() -> dict:
    avail = available_variants()
    return {meta["name"]: key for key, meta in avail.items()}


# ======================================================================
# Inference helpers
# ======================================================================
def preprocess_upload(img: Image.Image) -> np.ndarray:
    g = np.array(img.convert("L"))
    import cv2
    g = cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    g = g.astype(np.float32)
    if g.max() > 0:
        g = g / g.max()
    return g[np.newaxis, ..., np.newaxis]           # (1, 52, 52, 1)


def predict_probs(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x, verbose=0)[0]


def fig_probabilities(probs, classes, pal):
    order = np.argsort(probs)
    labels = [classes[i] for i in order]
    vals = probs[order]
    colors = ["#" + class_color(c, pal).lstrip("#") for c in labels]
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(classes) + 0.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(min(v + 0.015, 0.97), b.get_y() + b.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=9, color=pal["title"])
    ax.set_xlim(0, 1)
    ax.tick_params(colors=pal["muted"], labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color=pal["border"], alpha=0.5)
    fig.tight_layout()
    return fig


def fig_gauge(confidence: float, color: str, pal):
    fig, ax = plt.subplots(figsize=(2.2, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    theta_bg = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=14, color=pal["border"],
            solid_capstyle="round")
    theta = np.linspace(0, np.pi * confidence, 100)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color=color,
            solid_capstyle="round")
    ax.text(0, 0.12, f"{confidence:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=pal["title"])
    ax.text(0, -0.18, "confidence", ha="center", va="center",
            fontsize=9, color=pal["muted"])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.25)
    ax.axis("off")
    return fig


def run_predict(model, x, variant_key):
    probs = predict_probs(model, x)
    idx = int(np.argmax(probs))
    return probs, idx


# ======================================================================
# Layout
# ======================================================================
st.set_page_config(page_title="WaferDefect Studio", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

pal = PALETTES[st.session_state.get("palette", DEFAULT_PALETTE)]
st.markdown(build_css(pal), unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    new_pal = st.selectbox("🎨 Theme", list(PALETTES.keys()),
                           index=list(PALETTES.keys()).index(
                               st.session_state.get("palette", DEFAULT_PALETTE)))
    if new_pal != st.session_state.get("palette"):
        st.session_state["palette"] = new_pal
        st.rerun()
    pal = PALETTES[new_pal]

    st.markdown("## 🔬 WaferDefect Studio")
    st.caption("Diffusion augmentation · ViT · Grad-CAM")
    st.divider()
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()

    opts = variant_options()
    if len(opts) > 1:
        chosen = st.selectbox("Active model", list(opts.keys()))
        variant_key = opts[chosen]
    else:
        variant_key = list(opts.values())[0]
        meta = MODEL_VARIANTS[variant_key]
        if not os.path.exists(meta["path"]):
            st.warning("No trained model found — train first "
                       "(`python src/train.py`). Predictions disabled.")
    st.caption(f"Active: **{MODEL_VARIANTS[variant_key]['name']}** · "
               f"{IMG_SIZE}×{IMG_SIZE} · {len(CLASS_NAMES)} classes")

model = load_model(variant_key)

# ======================================================================
# Page: Predict
# ======================================================================
if page == PAGES[0]:
    st.markdown('<p class="gradient-title">Live Defect Prediction</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtle">Upload a wafer map — the active model '
                'classifies it and explains itself.</p>',
                unsafe_allow_html=True)
    st.divider()

    if model is None:
        st.error("No model checkpoint found. Train one first: "
                 "`python src/train.py` (or run `train_colab.ipynb`).")
        st.stop()

    up = st.file_uploader("Drop a wafer map image", type=["png", "jpg", "jpeg", "bmp"],
                          label_visibility="collapsed")
    c_demo1, c_demo2, _ = st.columns([1, 1, 2])
    demo_synthetic = c_demo1.button("🎲 Random synthetic map", use_container_width=True)
    demo_noise = c_demo2.button("🌫 Sample from diffusion", use_container_width=True)

    x = None
    title_img = None
    if up is not None:
        title_img = Image.open(up)
        x = preprocess_upload(title_img)
    elif demo_synthetic:
        rng = np.random.default_rng()
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        r = np.sqrt((xx - IMG_SIZE / 2) ** 2 + (yy - IMG_SIZE / 2) ** 2)
        kind = rng.integers(0, 4)
        if kind == 0:
            canvas[(r < 12) & (r > 2)] = 1
        elif kind == 1:
            canvas[r < 10] = 1
        elif kind == 2:
            canvas[r > IMG_SIZE / 2 - 4] = 1
        else:
            pts = rng.integers(4, IMG_SIZE - 4, (40, 2))
            canvas[pts[:, 0], pts[:, 1]] = 1
        x = canvas[np.newaxis, ..., np.newaxis]
        title_img = Image.fromarray((canvas * 255).astype(np.uint8))
    elif demo_noise and load_diffusion() is not None:
        from src.diffusion.sample import generate_samples
        samples = generate_samples(load_diffusion(), n=1)
        x = samples[:1]
        title_img = Image.fromarray((samples[0].squeeze() * 255).astype(np.uint8))

    if x is not None:
        probs, idx = run_predict(model, x, variant_key)
        pred = CLASS_NAMES[idx]
        conf = float(probs[idx])

        colL, colR = st.columns([1, 1.6], gap="large")
        with colL:
            st.markdown("#### Input")
            st.image(title_img, use_container_width=True, clamp=True)
            st.caption(f"{IMG_SIZE}×{IMG_SIZE} grayscale wafer map")
        with colR:
            sev = SEVERITY.get(pred, "info")
            ic = CLASS_ICONS.get(pred, "❓")
            cc = class_color(pred, pal)
            st.markdown(
                f"""<div class="card" style="border-color:{cc};
                box-shadow:0 0 24px {hex_to_rgba(cc,0.18)}">
                <span style="font-size:2rem">{ic}</span>
                <span style="font-size:1.7rem;font-weight:800;color:{cc};
                margin-left:10px">{pred}</span>
                <span class="badge" style="color:{SEVERITY_COLORS[sev]};
                border-color:{SEVERITY_COLORS[sev]};margin-left:12px">
                {sev.upper()}</span>
                <div class="subtle" style="margin-top:8px">
                {CLASS_INFO.get(pred, '')}</div>
                {'<div class="badge" style="color:#fbbf24;border-color:#fbbf24;margin-top:10px">MINORITY CLASS — diffusion-augmented in training</div>'
                 if pred in MINORITY_CLASSES_SET and variant_key == 'vit_diff' else ''}
                </div>""",
                unsafe_allow_html=True)
            g1, g2 = st.columns([1, 1.4])
            with g1:
                st.pyplot(fig_gauge(conf, cc, pal), use_container_width=True)
            with g2:
                st.pyplot(fig_probabilities(probs, CLASS_NAMES, pal),
                          use_container_width=True)

        # quick explanation preview
        st.divider()
        try:
            meta = MODEL_VARIANTS[variant_key]
            if meta["explain"] == "gradcam":
                heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=meta["layer"],
                                               pred_index=idx)
                overlay = overlay_gradcam(x[0], heatmap)
                h1, h2, h3 = st.columns(3)
                with h1:
                    st.image(x[0].squeeze(), caption="Input", use_container_width=True, clamp=True)
                with h2:
                    fig, ax = plt.subplots()
                    fig.patch.set_alpha(0)
                    ax.imshow(heatmap, cmap=pal["cmap"]); ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    st.caption("Grad-CAM heatmap")
                with h3:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                             caption="Overlay", use_container_width=True)
            else:
                from src.explain import attention_rollout_map
                roll = attention_rollout_map(model, x, variant_key)
                fig, ax = plt.subplots()
                fig.patch.set_alpha(0)
                ax.imshow(x[0].squeeze(), cmap="gray")
                ax.imshow(roll, cmap=pal["cmap"], alpha=0.55)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                st.caption("Attention rollout overlay (ViT)")
        except Exception as e:
            st.info(f"Explanation preview unavailable: {e}")

# ======================================================================
# Page: Diffusion Lab
# ======================================================================
elif page == PAGES[1]:
    st.markdown('<p class="gradient-title">Diffusion Lab</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtle">Class-conditional latent-diffusion sampler '
                '— the minority-class augmentation engine.</p>',
                unsafe_allow_html=True)
    st.divider()

    diff = load_diffusion()
    if diff is None:
        st.warning("No diffusion checkpoint found in "
                   f"`{DIFFUSION_CKPT_DIR}`. Train it with "
                   "`python src/train_diffusion.py` (Colab: `train_colab.ipynb`).")
        st.stop()

    cA, cB, cC = st.columns(3)
    cls = cA.selectbox("Condition class", CLASS_NAMES,
                       index=CLASS_NAMES.index("Scratch"))
    n = cB.slider("Samples", 4, 32, 8)
    steps = cC.slider("Denoising steps", 10, 200, 50, step=10)

    if st.button("✨ Generate", type="primary", use_container_width=True):
        from src.diffusion.sample import generate_samples
        with st.spinner("Sampling from latent diffusion …"):
            samples = generate_samples(diff, n=n, classes=[cls] * n, steps=steps)
        cols = st.columns(min(n, 8))
        for i in range(n):
            with cols[i % len(cols)]:
                st.image((samples[i].squeeze() * 255).astype(np.uint8),
                         caption=f"#{i+1}", use_container_width=True, clamp=True)
        buf = io.BytesIO()
        grid = np.concatenate([samples.squeeze(), np.ones(
            (samples.shape[0], 2, samples.shape[-1]), np.float32)], axis=1)
        Image.fromarray((grid * 255).astype(np.uint8)).save(buf, "PNG")
        st.download_button("⬇ Download grid PNG", buf.getvalue(),
                           f"diffusion_{cls}_{n}.png", use_container_width=True)

    st.divider()
    st.markdown("#### Why this matters")
    st.markdown(
        "Minority classes (Scratch, Donut, Near-full …) are under-represented in "
        "WM-811K, so classifiers overfit to majority patterns. We train a small "
        "**latent-diffusion model** on the training split only and inject synthetic "
        "minority maps during classifier training. The ablation (`📊 Experiments`) "
        "quantifies the gain."
    )

# ======================================================================
# Page: Experiments
# ======================================================================
elif page == PAGES[2]:
    st.markdown('<p class="gradient-title">Experiments — 3-Arm Ablation</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtle">CNN baseline vs. Lightweight ViT vs. '
                'ViT + diffusion augmentation. Trained on WM-811K.</p>',
                unsafe_allow_html=True)
    st.divider()

    df = load_csv_safe(os.path.join(OUT_ROOT, "ablation_results.csv"))
    if df is None:
        st.info("No results yet — run `python src/train.py` to produce "
                "`outputs/ablation_results.csv`.")
    else:
        met = st.columns(min(len(df), 3))
        for c, (_, row) in zip(met, df.iterrows()):
            with c:
                key = row.get("variant", "")
                color = MODEL_VARIANTS.get(key, {}).get("color", pal["accents"][0])
                st.markdown(f"#### {row.get('name', key)}")
                st.metric("Test Accuracy", f"{row.get('test_acc', 0):.1%}")
                st.metric("Macro F1", f"{row.get('macro_f1', 0):.3f}")
                if "params" in row:
                    st.metric("Params", f"{int(row['params']):,}")
        st.divider()
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇ Export CSV", df.to_csv(index=False).encode(),
                           "ablation_results.csv", use_container_width=True)

    # per-class F1 comparison if present
    df_pc = load_csv_safe(os.path.join(OUT_ROOT, "per_class_f1.csv"))
    if df_pc is not None:
        st.divider()
        st.markdown("#### Per-class F1 by variant")
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        x_np = np.arange(len(df_pc))
        width = 0.8 / max(len(df_pc.columns) - 1, 1)
        for j, col in enumerate([c for c in df_pc.columns if c != "class"]):
            ax.bar(x_np + j * width, df_pc[col], width,
                   label=col, color=pal["accents"][j % len(pal["accents"])])
        ax.set_xticks(x_np + width)
        ax.set_xticklabels(df_pc["class"], rotation=30, ha="right")
        ax.tick_params(colors=pal["muted"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, labelcolor=pal["muted"])
        st.pyplot(fig, use_container_width=True)

# ======================================================================
# Page: Cross-Dataset
# ======================================================================
elif page == PAGES[3]:
    st.markdown('<p class="gradient-title">Cross-Dataset Robustness</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtle">Train on one corpus, test zero-shot on '
                'another. Robustness is the headline contribution.</p>',
                unsafe_allow_html=True)
    st.divider()

    df = load_csv_safe(os.path.join(OUT_ROOT, "cross_dataset_results.csv"))
    if df is None:
        st.info("No cross-dataset results yet — run `python src/evaluate.py "
                "--cross` after training.")
    else:
        pivot_cols = [c for c in df.columns if c not in ("train_on", "test_on")]
        try:
            piv = df.pivot_table(index="train_on", columns="test_on",
                                 values="macro_f1", aggfunc="first")
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            im = ax.imshow(piv.values, cmap=pal["cmap"], vmin=0, vmax=1)
            ax.set_xticks(range(len(piv.columns)), piv.columns)
            ax.set_yticks(range(len(piv.index)), piv.index)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=11,
                            fontweight="bold")
            ax.set_title("Macro-F1: train → test", color=pal["title"])
            fig.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not pivot results: {e}")
        st.dataframe(df, use_container_width=True)

# ======================================================================
# Page: Explainability
# ======================================================================
elif page == PAGES[4]:
    st.markdown('<p class="gradient-title">Explainability Gallery</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtle">Grad-CAM (CNN) vs. attention rollout (ViT) '
                '— where each architecture looks when it votes.</p>',
                unsafe_allow_html=True)
    st.divider()

    grid_files = sorted(glob.glob(os.path.join(OUT_ROOT, "explain", "*.png")))
    if not grid_files:
        st.info("No explanation exports yet — run `python src/evaluate.py "
                "--explain` after training.")
    else:
        per_row = 3
        for i in range(0, len(grid_files), per_row):
            cols = st.columns(per_row)
            for j, f in enumerate(grid_files[i:i + per_row]):
                with cols[j]:
                    st.image(Image.open(f), use_container_width=True)
                    st.caption(os.path.basename(f).replace("_", " "))

    st.divider()
    up2 = st.file_uploader("Explain a custom map now", type=["png", "jpg", "bmp"],
                           key="explain_upload")
    if up2 is not None and model is not None:
        x = preprocess_upload(Image.open(up2))
        meta = MODEL_VARIANTS[variant_key]
        try:
            if meta["explain"] == "gradcam":
                hm = make_gradcam_heatmap(x, model, last_conv_layer_name=meta["layer"])
                st.image(overlay_gradcam(x[0], hm), use_container_width=True,
                         clamp=True)
            else:
                from src.explain import attention_rollout_map
                st.image(attention_rollout_map(model, x, variant_key),
                         use_container_width=True, clamp=True)
        except Exception as e:
            st.warning(f"Explanation failed: {e}")

# ======================================================================
# Page: About
# ======================================================================
else:
    st.markdown('<p class="gradient-title">About WaferDefect Studio</p>',
                unsafe_allow_html=True)
    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("#### Research question")
        st.markdown(
            "Do diffusion-synthesized minority-class wafer maps improve a "
            "lightweight transformer's *cross-dataset* robustness on mixed-type "
            "defect detection — and can the predictions be trusted via "
            "explainability maps?")
        st.markdown("#### Contributions")
        st.markdown(
            "1. **Latent-diffusion minority-class augmentation** (SD-style, "
            "trained from scratch, no external checkpoint)\n"
            "2. **Lightweight ViT** (~500K params) benchmarked against a CNN\n"
            "3. **Cross-dataset robustness protocol** WM-811K → MixedWM38 with "
            "Grad-CAM / attention-rollout evaluation")
    with b:
        st.markdown("#### Datasets")
        st.markdown(
            "- **WM-811K** — 811,457 labeled maps (train)\n"
            "- **MixedWM38** — 38,015 mixed-type maps (zero-shot test)\n"
            "- **Sample-WaferMap** — auxiliary benchmark")
        st.markdown("#### Stack")
        st.code("TensorFlow/Keras · Streamlit · NumPy · OpenCV", language="bash")
        st.markdown("#### Train from scratch")
        st.code(
            "pip install -r requirements.txt\n"
            "python src/train_diffusion.py      # latent diffusion\n"
            "python src/train.py                # 3 classifier arms\n"
            "python src/evaluate.py --all       # ablation + cross-dataset + explain",
            language="bash")
