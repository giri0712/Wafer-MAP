"""
app.py
Streamlit demo UI for WM-811K Wafer Map Pattern Classification
with Grad-CAM explainability.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import tensorflow as tf
from PIL import Image

# Local
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import IMG_SIZE, DEFECT_CLASSES
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "outputs/best_model.keras"
CLASSES_PATH = "outputs/label_classes.npy"

PATTERN_INFO = {
    "Center":    ("🔴", "#ef4444", "Defects clustered at the wafer center. Often caused by chuck contamination or spin-coat center anomalies."),
    "Donut":     ("🟠", "#f97316", "Ring-shaped defect band inside the edge ring. Linked to focus or exposure non-uniformity."),
    "Edge-Loc":  ("🟡", "#eab308", "Localized defects at the wafer edge. Common with edge bead removal issues."),
    "Edge-Ring": ("🔶", "#f59e0b", "Full peripheral ring of defects. Typically indicates EBR or CMP edge problems."),
    "Loc":       ("🟢", "#22c55e", "Isolated local cluster of defects. Usually particle contamination or mask defect."),
    "Near-full": ("⚫", "#6b7280", "Nearly all dies failing. Catastrophic process failure — check entire process step."),
    "Random":    ("🔵", "#3b82f6", "No spatial pattern detected. May indicate random particle events or minor process noise."),
    "Scratch":   ("🟣", "#a855f7", "Linear trail of defects. Physical scratch from handling, chuck, or probe damage."),
    "none":      ("✅", "#00d4b4", "No defect pattern detected. Wafer appears clean with high yield."),
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


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model ...")
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model   = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True).tolist() \
              if os.path.exists(CLASSES_PATH) else DEFECT_CLASSES
    return model, classes


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Resize + normalize → (1, 64, 64, 1)"""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    return img[np.newaxis, ..., np.newaxis]  # (1, 64, 64, 1)


def severity_color(sev: str) -> str:
    return {"critical": "#ef4444", "warning": "#f59e0b", "info": "#3b82f6"}.get(sev, "#6b7280")


def render_top3_chart(probs: np.ndarray, classes: list):
    top3_idx  = np.argsort(probs)[::-1][:3]
    top3_cls  = [classes[i] for i in top3_idx]
    top3_prob = [probs[i] for i in top3_idx]
    colors    = [PATTERN_INFO.get(c, ("", "#6b7280", ""))[1] for c in top3_cls]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    bars = ax.barh(top3_cls[::-1], top3_prob[::-1], color=colors[::-1], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Top-3 Predictions", fontsize=10)
    for bar, prob in zip(bars, top3_prob[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Page Layout ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Map Pattern Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Wafer Map Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Analyze", "📚 Pattern Library", "ℹ️ About"])
    st.markdown("---")
    st.caption("WM-811K Dataset · TensorFlow/Keras · Grad-CAM")

# ── Load Model ────────────────────────────────────────────────────────────────
model, classes = load_model_and_classes()

# ── Page: Analyze ─────────────────────────────────────────────────────────────
if page == "🏠 Analyze":
    st.title("🔬 Wafer Map Defect Pattern Classifier")
    st.markdown("Upload a wafer map image to classify its defect pattern using the trained CNN model.")

    if model is None:
        st.error(
            "**Model not found.** Please train the model first:\n\n"
            "```bash\npython src/train.py\npython src/evaluate.py\n```"
        )
        st.stop()

    uploaded = st.file_uploader(
        "Drop a wafer map image here",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Accepts grayscale or color wafer map images up to 20MB"
    )

    if uploaded:
        # Load image
        pil_img  = Image.open(uploaded).convert("L")
        img_arr  = np.array(pil_img)
        img_proc = preprocess_image(img_arr)

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(pil_img, caption="Original (grayscale)", use_container_width=True)

        with col2:
            st.subheader("🧠 Analysis Result")

            with st.spinner("Running inference ..."):
                probs      = model.predict(img_proc, verbose=0)[0]
                pred_idx   = int(np.argmax(probs))
                pred_label = classes[pred_idx]
                confidence = float(probs[pred_idx])

            icon, color, desc = PATTERN_INFO.get(pred_label, ("❓", "#6b7280", "Unknown pattern."))
            sev = SEVERITY.get(pred_label, "info")

            # Pattern badge
            st.markdown(
                f"""<div style='background:{color}22;border:2px solid {color};border-radius:12px;
                padding:16px 20px;margin-bottom:16px;'>
                <span style='font-size:28px'>{icon}</span>
                <span style='font-size:22px;font-weight:700;color:{color};margin-left:10px'>
                {pred_label}</span>
                <span style='font-size:12px;color:{severity_color(sev)};margin-left:12px;
                font-weight:600;text-transform:uppercase'>{sev}</span>
                </div>""",
                unsafe_allow_html=True
            )

            # Confidence meter
            st.markdown(f"**Confidence: {confidence:.1%}**")
            st.progress(confidence)

            # Description
            st.info(f"📋 {desc}")

            # Top-3 chart
            fig_top3 = render_top3_chart(probs, classes)
            st.pyplot(fig_top3, use_container_width=False)

        # Grad-CAM section
        st.divider()
        st.subheader("🔍 Grad-CAM Explainability")

        try:
            heatmap  = make_gradcam_heatmap(img_proc, model, last_conv_layer_name="conv6",
                                             pred_index=pred_idx)
            overlay  = overlay_gradcam(img_proc[0], heatmap)

            gc_col1, gc_col2, gc_col3 = st.columns(3)
            with gc_col1:
                st.image(img_proc[0].squeeze(), caption="Preprocessed (64×64)",
                         use_container_width=True, clamp=True)
            with gc_col2:
                fig_hm, ax = plt.subplots()
                ax.imshow(heatmap, cmap="jet")
                ax.axis("off")
                st.pyplot(fig_hm, use_container_width=True)
                st.caption("Activation Heatmap")
            with gc_col3:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                         caption="Grad-CAM Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM visualization unavailable: {e}")

# ── Page: Pattern Library ─────────────────────────────────────────────────────
elif page == "📚 Pattern Library":
    st.title("📚 Defect Pattern Library")
    st.markdown("Reference guide for all 9 WM-811K defect pattern classes.")
    st.divider()

    cols = st.columns(3)
    for i, (cls, (icon, color, desc)) in enumerate(PATTERN_INFO.items()):
        with cols[i % 3]:
            st.markdown(
                f"""<div style='background:{color}15;border:1.5px solid {color}50;
                border-radius:10px;padding:14px;margin-bottom:14px;min-height:120px'>
                <div style='font-size:22px'>{icon} <b style='color:{color}'>{cls}</b></div>
                <div style='font-size:13px;color:#ccc;margin-top:8px'>{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )

# ── Page: About ───────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("""
    ### Wafer Map Pattern Intelligence

    This tool uses a **Convolutional Neural Network (CNN)** trained on the
    **WM-811K dataset** to classify semiconductor wafer map defect patterns.

    **Dataset:** WM-811K — 811,457 wafer maps across 9 defect classes  
    **Model:** Custom CNN with 6 Conv layers + BatchNorm + Dropout  
    **Explainability:** Grad-CAM heatmaps highlight regions driving predictions  

    **Stack:**
    - Python 3.8+ · TensorFlow/Keras  
    - NumPy · Pandas · OpenCV  
    - Matplotlib · Seaborn  
    - Streamlit  

    **Dataset source:** [Kaggle — WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
    """)
