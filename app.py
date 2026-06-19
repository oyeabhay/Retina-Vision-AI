"""
Human Eye Disease Prediction System
Streamlit Web Application

Run:
    streamlit run app.py
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Disease Prediction System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Imports that depend on TF (lazy to keep startup fast)
# ─────────────────────────────────────────────
import tensorflow as tf

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = os.path.join("model", "best_model.keras")
IDX_PATH    = os.path.join("model", "class_indices.json")

DISEASE_INFO = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "description": (
            "Abnormal growth of blood vessels from the choroid layer beneath the "
            "retina. These vessels can leak fluid and blood, leading to scarring "
            "and central vision loss."
        ),
        "symptoms": [
            "Blurry or distorted central vision",
            "Straight lines appear wavy (metamorphopsia)",
            "Dark spot in central vision",
            "Reduced color perception",
        ],
        "treatment": "Anti-VEGF injections (ranibizumab, bevacizumab), photodynamic therapy.",
        "urgency": "🔴 High – Consult an ophthalmologist immediately.",
        "color": "#FF6B6B",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "description": (
            "Fluid accumulation in the macula due to leaking retinal blood vessels "
            "caused by diabetic retinopathy. One of the leading causes of vision "
            "impairment in working-age adults worldwide."
        ),
        "symptoms": [
            "Blurry or fluctuating vision",
            "Double vision",
            "Floaters",
            "Difficulty reading",
        ],
        "treatment": "Anti-VEGF injections, laser photocoagulation, corticosteroid implants.",
        "urgency": "🔴 High – Requires prompt medical attention.",
        "color": "#FFA07A",
    },
    "DRUSEN": {
        "full_name": "Drusen (Early AMD)",
        "description": (
            "Small yellow or white deposits of lipids and proteins that accumulate "
            "under the retinal pigment epithelium. Drusen are an early hallmark "
            "of age-related macular degeneration (AMD)."
        ),
        "symptoms": [
            "Mild blurring of central vision",
            "Difficulty adapting to low light",
            "Slightly washed-out color vision",
        ],
        "treatment": "AREDS2 supplements, lifestyle modifications, regular monitoring.",
        "urgency": "🟡 Moderate – Regular follow-up with an eye specialist recommended.",
        "color": "#FFD700",
    },
    "NORMAL": {
        "full_name": "Normal Retina",
        "description": (
            "No signs of retinal disease detected. The OCT scan shows a healthy "
            "retinal structure with well-defined layers and no fluid accumulation "
            "or abnormal deposits."
        ),
        "symptoms": [],
        "treatment": "Continue routine eye check-ups (annually or as advised).",
        "urgency": "🟢 No immediate action required.",
        "color": "#90EE90",
    },
}

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Top banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #0d3b6e 50%, #1a1f35 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2a4a7f;
    text-align: center;
}
.hero-banner h1 { color: #4fc3f7; font-size: 2.2rem; margin: 0; }
.hero-banner p  { color: #b0bec5; font-size: 1rem; margin-top: .4rem; }

/* Result card */
.result-card {
    border-radius: 12px;
    padding: 1.4rem;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1rem;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1e2130;
    border-radius: 8px;
    height: 22px;
    margin: 4px 0 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
    transition: width .5s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #111827; }

/* Upload zone */
.upload-zone {
    border: 2px dashed #2a4a7f;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #0d1626;
}

/* Info pills */
.pill {
    display: inline-block;
    background: rgba(79, 195, 247, 0.15);
    color: #4fc3f7;
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(IDX_PATH):
        with open(IDX_PATH) as f:
            idx_to_class = json.load(f)
    else:
        idx_to_class = {str(i): c for i, c in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"])}
    return model, idx_to_class


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, idx_to_class, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    results = [
        {"class": idx_to_class[str(i)], "confidence": float(preds[i])}
        for i in range(len(preds))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def confidence_bar(label, confidence, color):
    pct = int(confidence * 100)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
      <span style="width:70px;color:#ccc;font-size:0.85rem">{label}</span>
      <div class="conf-bar-bg" style="flex:1">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color}">
          {pct}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ About This App")
    st.markdown("""
    This system uses a **MobileNetV2** deep learning model trained on
    **OCT (Optical Coherence Tomography)** retinal scans to detect:
    """)

    for code, info in DISEASE_INFO.items():
        st.markdown(
            f'<span class="pill">{code}</span> {info["full_name"]}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Architecture | MobileNetV2 |
    | Input Size | 224 × 224 |
    | Classes | 4 |
    | Training Data | Kaggle OCT 2017 |
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This tool is for **educational purposes only** and does NOT replace "
        "professional medical diagnosis. Always consult a qualified ophthalmologist."
    )

    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [Kaggle OCT Dataset](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct)
    - [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
    """)


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>👁️ Human Eye Disease Prediction System</h1>
  <p>Upload a retinal OCT scan to detect CNV · DME · Drusen · Normal</p>
</div>
""", unsafe_allow_html=True)

model, idx_to_class = load_model()

if model is None:
    st.error(
        "⚠️ **Trained model not found.**\n\n"
        "Please run the training script first:\n"
        "```bash\n"
        "python train_model.py --data_dir OCT2017 --epochs 10\n"
        "```\n\n"
        "Then place `best_model.keras` and `class_indices.json` inside the `model/` folder."
    )
    st.stop()

# ── Upload section ──────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload OCT Image")
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a grayscale or color OCT retinal scan.",
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        with col_preview:
            st.markdown("### 🔍 Preview")
            st.image(img, caption=f"{uploaded.name} ({img.size[0]}×{img.size[1]} px)",
                     use_container_width=True)

    else:
        with col_preview:
            st.markdown("### 🔍 Preview")
            st.info("Image preview will appear here after upload.")

# ── Prediction ──────────────────────────────
if uploaded:
    with st.spinner("Analysing retinal scan…"):
        arr     = preprocess_image(img)
        results = predict(model, idx_to_class, arr)

    top         = results[0]
    top_class   = top["class"]
    top_conf    = top["confidence"]
    info        = DISEASE_INFO.get(top_class, {})
    badge_color = info.get("color", "#4fc3f7")

    st.markdown("---")
    st.markdown("## 🩺 Diagnosis Result")

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Primary prediction card
        st.markdown(f"""
        <div class="result-card" style="background:linear-gradient(135deg,{badge_color}22,{badge_color}08);
             border-color:{badge_color}55">
          <div style="font-size:2.5rem;text-align:center">{
            "🔴" if top_class in ("CNV","DME") else "🟡" if top_class=="DRUSEN" else "🟢"
          }</div>
          <h2 style="color:{badge_color};text-align:center;margin:.3rem 0">{top_class}</h2>
          <p style="color:#b0bec5;text-align:center;font-size:0.9rem">{info.get("full_name","")}</p>
          <hr style="border-color:{badge_color}44">
          <p style="color:#e0e0e0;font-size:0.9rem">{info.get("description","")}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Confidence Scores")
        bar_colors = {"CNV": "#FF6B6B", "DME": "#FFA07A",
                      "DRUSEN": "#FFD700", "NORMAL": "#90EE90"}
        for r in results:
            confidence_bar(r["class"], r["confidence"],
                           bar_colors.get(r["class"], "#4fc3f7"))

    with res_right:
        if info.get("symptoms"):
            st.markdown("#### 🩻 Common Symptoms")
            for s in info["symptoms"]:
                st.markdown(f"• {s}")

        st.markdown("#### 💊 Treatment Options")
        st.info(info.get("treatment", "Consult a specialist."))

        st.markdown("#### 🚨 Urgency Level")
        st.markdown(info.get("urgency", ""))

        # Confidence gauge
        st.markdown("#### 🎯 Top Prediction Confidence")
        st.progress(top_conf)
        st.markdown(
            f"<p style='text-align:center;color:{badge_color};font-size:1.4rem;"
            f"font-weight:bold'>{top_conf*100:.1f}%</p>",
            unsafe_allow_html=True,
        )

    # Low confidence warning
    if top_conf < 0.60:
        st.warning(
            "⚠️ The model's confidence is **below 60 %**. "
            "The image quality may be low, or this could be an atypical case. "
            "Please consult an ophthalmologist for a definitive diagnosis."
        )

# ── Disease reference cards ─────────────────
st.markdown("---")
st.markdown("## 📚 Disease Reference Guide")

cols = st.columns(4)
for i, (code, info) in enumerate(DISEASE_INFO.items()):
    with cols[i]:
        st.markdown(f"""
        <div class="result-card" style="background:#131928;border-color:{info['color']}44;
             min-height:180px">
          <h4 style="color:{info['color']};margin-top:0">{code}</h4>
          <p style="color:#9e9e9e;font-size:0.78rem">{info['full_name']}</p>
          <p style="color:#ccc;font-size:0.82rem">{info['description'][:150]}…</p>
        </div>
        """, unsafe_allow_html=True)
