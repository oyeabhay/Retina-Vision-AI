"""
Retina Vision AI — Human Eye Disease Prediction System
Streamlit Web Application

Run:
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import numpy as np
from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retina Vision AI",
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
IMG_SIZE  = (224, 224)
MODEL_PATH = os.path.join("model", "best_model.keras")
IDX_PATH   = os.path.join("model", "class_indices.json")

APP_NAME      = "Retina Vision AI"
MODEL_BADGE   = "MOBILENETV2"
DATASET_BADGE = "OCT 2017 DATASET"
AVATAR_TEXT   = "DR"

# color values are *semantic keys* resolved to CSS variables at render time
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
        "urgency_short": "High urgency",
        "color": "red",
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
        "urgency_short": "High urgency",
        "color": "orange",
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
        "urgency_short": "Moderate",
        "color": "yellow",
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
        "urgency_short": "No action needed",
        "color": "green",
    },
}

NAV_ITEMS = [
    ("scan",    "🔬", "Scan & Predict"),
    ("reports", "📋", "My Reports"),
    ("guide",   "📖", "Disease Guide"),
    ("how",     "⚙️", "How It Works"),
]

VAR    = {"red": "var(--red)",      "orange": "var(--orange)",      "yellow": "var(--yellow)",      "green": "var(--green)",      "teal": "var(--teal)"}
VARSFT = {"red": "var(--red-soft)", "orange": "var(--orange-soft)", "yellow": "var(--yellow-soft)", "green": "var(--green-soft)", "teal": "var(--teal-soft)"}


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "scan"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "history" not in st.session_state:
    st.session_state.history = []
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None


# ─────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────
def get_theme(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#070d0a", bg_sidebar="#0a130d", card="#0f1c16", card_alt="#0c1712",
            border="rgba(255,255,255,0.08)", border_strong="rgba(255,255,255,0.16)",
            text="#eef5f1", text_secondary="#93a39b", text_muted="#5f6f67",
            teal="#2dd9a3", teal_soft="rgba(45,217,163,0.12)",
            red="#ff6b7a", red_soft="rgba(255,107,122,0.12)",
            orange="#ffac4b", orange_soft="rgba(255,172,75,0.12)",
            yellow="#ffd23f", yellow_soft="rgba(255,210,63,0.12)",
            green="#5fe3a1", green_soft="rgba(95,227,161,0.12)",
        )
    return dict(
        bg="#f4f7f5", bg_sidebar="#ffffff", card="#ffffff", card_alt="#f7faf8",
        border="rgba(15,30,22,0.08)", border_strong="rgba(15,30,22,0.16)",
        text="#16201b", text_secondary="#51625a", text_muted="#84948c",
        teal="#0c9d6f", teal_soft="rgba(12,157,111,0.10)",
        red="#d8334a", red_soft="rgba(216,51,74,0.10)",
        orange="#c97412", orange_soft="rgba(201,116,18,0.10)",
        yellow="#b08a0f", yellow_soft="rgba(176,138,15,0.10)",
        green="#149562", green_soft="rgba(20,149,98,0.10)",
    )


def inject_css():
    T = get_theme(st.session_state.dark_mode)

    root_vars = f"""
    :root {{
      --bg: {T['bg']};
      --bg-sidebar: {T['bg_sidebar']};
      --card: {T['card']};
      --card-alt: {T['card_alt']};
      --border: {T['border']};
      --border-strong: {T['border_strong']};
      --text: {T['text']};
      --text-secondary: {T['text_secondary']};
      --text-muted: {T['text_muted']};
      --teal: {T['teal']};
      --teal-soft: {T['teal_soft']};
      --red: {T['red']};
      --red-soft: {T['red_soft']};
      --orange: {T['orange']};
      --orange-soft: {T['orange_soft']};
      --yellow: {T['yellow']};
      --yellow-soft: {T['yellow_soft']};
      --green: {T['green']};
      --green-soft: {T['green_soft']};
    }}
    """

    static_css = """
    html, body, [data-testid="stAppViewContainer"], .main {
        background: var(--bg) !important;
        color: var(--text);
    }
    [data-testid="stHeader"] { background: transparent; }
    div.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

    .sidebar-logo { display:flex; align-items:center; gap:.7rem; margin-bottom:1.6rem; }
    .logo-icon {
        width:42px; height:42px; border-radius:12px; flex-shrink:0;
        background: linear-gradient(135deg, var(--teal) 0%, var(--yellow) 100%);
        display:flex; align-items:center; justify-content:center; font-size:1.3rem;
        box-shadow: 0 0 16px rgba(45,217,163,0.35);
    }
    .logo-title { font-size:1.15rem; font-weight:800; color:var(--text); line-height:1.2; }
    .logo-sub   { font-size:.68rem; letter-spacing:.06em; color:var(--text-muted); font-weight:600; }

    .nav-label {
        font-size:.7rem; letter-spacing:.1em; color:var(--text-muted);
        font-weight:700; margin: 0 0 .5rem .2rem;
    }

    section[data-testid="stSidebar"] .stButton button {
        display:flex !important;
        align-items:center !important;
        justify-content:flex-start !important;
        text-align:left !important;
        border-radius:8px !important;
        padding:.5rem .7rem !important;
        font-weight:500 !important;
        margin-bottom:.15rem;
    }
    section[data-testid="stSidebar"] .stButton button div,
    section[data-testid="stSidebar"] .stButton button span,
    section[data-testid="stSidebar"] .stButton button p {
        text-align:left !important;
        justify-content:flex-start !important;
        width:100% !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background:transparent !important;
        color:var(--text-secondary) !important;
        border:none !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background:var(--card-alt) !important;
        color:var(--text) !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background:var(--teal-soft) !important;
        color:var(--teal) !important;
        border:none !important;
        border-left:3px solid var(--teal) !important;
        box-shadow:none !important;
    }

    .info-box {
        margin-top:1.4rem; padding:.8rem .9rem; border-radius:10px;
        background: var(--orange-soft); border:1px solid var(--orange);
        color: var(--text-secondary); font-size:.78rem; line-height:1.45;
    }
    .info-box b { color: var(--orange); }

    /* ---------- Header row ---------- */
    .page-title { font-size:1.3rem; font-weight:700; color:var(--text); padding-top:.35rem; }
    .model-badge {
        display:inline-block; padding:.35rem .9rem; border-radius:20px;
        border:1px solid var(--teal); color:var(--teal); background:var(--teal-soft);
        font-size:.72rem; font-weight:700; letter-spacing:.04em; white-space:nowrap;
        text-align:center; margin-top:.2rem;
    }
    .avatar {
        width:38px; height:38px; border-radius:50%;
        background: linear-gradient(135deg, var(--teal) 0%, var(--yellow) 100%);
        color:#06140d; display:flex; align-items:center; justify-content:center;
        font-weight:800; font-size:.85rem; margin-left:auto; margin-top:.05rem;
        box-shadow: 0 0 10px rgba(45,217,163,0.25);
    }

    /* ---------- Hero banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, var(--card) 0%, var(--card-alt) 100%);
        border: 1px solid var(--border-strong);
        border-radius: 16px; padding: 1.6rem 2rem; margin: 1.2rem 0 1.6rem;
        display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
    }
    .hero-left { display:flex; align-items:center; gap:1.1rem; }
    .hero-icon {
        width:62px; height:62px; border-radius:16px; flex-shrink:0;
        background: linear-gradient(135deg, var(--teal) 0%, var(--yellow) 100%);
        display:flex; align-items:center; justify-content:center; font-size:1.9rem;
        box-shadow: 0 0 24px rgba(45,217,163,0.3);
    }
    .hero-title { font-size:1.7rem; font-weight:800; color:var(--text); }
    .hero-sub   { color:var(--text-secondary); font-size:.92rem; margin-top:.15rem; }
    .hero-badge {
        border:1px solid var(--orange); color:var(--orange); background:var(--orange-soft);
        padding:.45rem 1rem; border-radius:20px; font-size:.74rem; font-weight:700;
        letter-spacing:.04em; white-space:nowrap;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size:.76rem; font-weight:700; letter-spacing:.08em;
        color:var(--teal); text-transform:uppercase; margin-bottom:.7rem;
    }

    /* ---------- Generic card ---------- */
    .card {
        background:var(--card); border:1px solid var(--border-strong);
        border-radius:14px; padding:1.3rem 1.5rem;
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--card-alt) !important;
        border:2px dashed var(--border-strong) !important;
        border-radius:14px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--teal) !important; color:#06140d !important;
        border:none !important; border-radius:8px !important; font-weight:700 !important;
    }
    div[data-testid="stFileUploader"] section { background: transparent; }

    .preview-empty {
        background:var(--card-alt); border:1px solid var(--border);
        border-radius:14px; min-height:220px; display:flex; align-items:center;
        justify-content:center; color:var(--text-muted); font-size:.95rem;
    }

    /* ---------- Result card ---------- */
    .result-badge {
        display:inline-block; padding:.4rem .9rem; border-radius:8px;
        font-weight:800; font-size:.95rem; letter-spacing:.02em;
    }
    .result-title { font-size:1.4rem; font-weight:800; color:var(--text); margin:.6rem 0 .15rem; }
    .result-sub   { color:var(--text-muted); font-size:.85rem; margin-bottom:.9rem; }
    .result-desc  { color:var(--text-secondary); font-size:.92rem; line-height:1.55; }

    /* ---------- Confidence bars ---------- */
    .conf-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
    .conf-label { width:78px; flex-shrink:0; color:var(--text-secondary); font-size:.82rem; font-weight:600; }
    .conf-bar-bg { flex:1; background:var(--card-alt); border-radius:8px; height:24px; overflow:hidden; }
    .conf-bar-fill {
        height:100%; border-radius:8px; display:flex; align-items:center;
        padding-left:10px; font-size:.76rem; font-weight:700; color:#06140d;
        transition: width .5s ease;
    }

    /* ---------- Symptoms / treatment / urgency ---------- */
    .symptom-list { margin:0 0 .2rem 0; padding-left:1.1rem; color:var(--text); }
    .symptom-list li { color:var(--text); font-size:.9rem; margin-bottom:.45rem; padding-left:.2rem; }
    .symptom-list li::marker { color:var(--orange); }

    .treat-box, .urgency-box {
        border-radius:10px; padding:.85rem 1rem; font-size:.88rem; font-weight:500;
        border:1px solid; line-height:1.5;
    }

    .stat-card {
        background:var(--card-alt); border:1px solid var(--border-strong);
        border-radius:14px; padding:1.4rem; text-align:center; margin-top:1rem;
    }
    .stat-number { font-size:2.6rem; font-weight:800; line-height:1; }
    .stat-label  { color:var(--text-muted); font-size:.75rem; letter-spacing:.08em; font-weight:700; margin-top:.4rem; }

    .divider { border:none; border-top:1px solid var(--border); margin:2rem 0; }

    /* ---------- Reference cards ---------- */
    .ref-card {
        background:var(--card); border:1px solid var(--border-strong);
        border-radius:14px; padding:1.1rem 1.2rem; min-height:230px;
    }
    .ref-code { font-size:1.05rem; font-weight:800; margin-bottom:.2rem; }
    .ref-name { color:var(--text-muted); font-size:.78rem; margin-bottom:.6rem; }
    .ref-desc { color:var(--text-secondary); font-size:.82rem; line-height:1.5; margin-bottom:.7rem; }
    .ref-urgency { font-size:.8rem; font-weight:700; }

    /* ---------- My Reports ---------- */
    .report-row {
        display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.4rem;
        background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:.8rem 1.1rem; margin-bottom:.6rem;
    }
    .report-class { font-weight:800; margin-right:.7rem; }
    .report-file  { color:var(--text-secondary); font-size:.88rem; }
    .report-meta  { color:var(--text-muted); font-size:.8rem; display:flex; gap:1.2rem; }

    /* ---------- How it works ---------- */
    .step-num {
        width:38px; height:38px; border-radius:50%; flex-shrink:0;
        background:var(--teal-soft); color:var(--teal); font-weight:800;
        display:flex; align-items:center; justify-content:center; font-size:1.05rem;
    }
    """

    st.markdown(f"<style>{root_vars}{static_css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model helpers
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


def confidence_bar(label, confidence, color_var):
    pct = int(round(confidence * 100))
    st.markdown(f"""
    <div class="conf-row">
      <span class="conf-label">{label}</span>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color_var}">{pct}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Shared UI pieces
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo">
          <div class="logo-icon">👁️</div>
          <div>
            <div class="logo-title">{APP_NAME}</div>
            <div class="logo-sub">AI · VISION HEALTH</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='nav-label'>NAVIGATION</div>", unsafe_allow_html=True)
        for key, icon, label in NAV_ITEMS:
            active = st.session_state.page == key
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<div class='nav-label' style='margin-top:1.8rem'>INFO</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          For <b>educational use only</b>. Does not replace a professional
          ophthalmologist's diagnosis.
        </div>
        """, unsafe_allow_html=True)


def render_header(title: str, icon: str):
    c1, c2, c3, c4 = st.columns([5, 2.1, 1.4, 0.7])
    with c1:
        st.markdown(f"<div class='page-title'>{icon} {title}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='model-badge'>{MODEL_BADGE}</div>", unsafe_allow_html=True)
    with c3:
        toggled = st.toggle("Dark", value=st.session_state.dark_mode, key="dark_toggle")
        if toggled != st.session_state.dark_mode:
            st.session_state.dark_mode = toggled
            st.rerun()
    with c4:
        st.markdown(f"<div class='avatar'>{AVATAR_TEXT}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────
def scan_predict_page():
    render_header("Scan & Predict", "🔬")

    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-left">
        <div class="hero-icon">👁️</div>
        <div>
          <div class="hero-title">{APP_NAME}</div>
          <div class="hero-sub">Upload an OCT retinal scan to screen for CNV · DME · Drusen · Normal</div>
        </div>
      </div>
      <div class="hero-badge">{DATASET_BADGE}</div>
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
        return

    col_upload, col_preview = st.columns(2, gap="large")

    with col_upload:
        st.markdown("<div class='section-label'>📤 Upload OCT Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a grayscale or color OCT retinal scan.",
            label_visibility="collapsed",
        )

    img = None
    with col_preview:
        st.markdown("<div class='section-label'>🔍 Preview</div>", unsafe_allow_html=True)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption=f"{uploaded.name} ({img.size[0]}×{img.size[1]} px)",
                      use_container_width=True)
        else:
            st.markdown("<div class='preview-empty'>Image preview appears here</div>",
                        unsafe_allow_html=True)

    if uploaded and img is not None:
        with st.spinner("Analysing retinal scan…"):
            arr = preprocess_image(img)
            results = predict(model, idx_to_class, arr)

        top = results[0]
        top_class, top_conf = top["class"], top["confidence"]
        info = DISEASE_INFO.get(top_class, {})
        color_key = info.get("color", "teal")
        color, color_soft = VAR[color_key], VARSFT[color_key]

        # log to history once per unique upload
        upload_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.last_upload_id != upload_id:
            st.session_state.history.insert(0, {
                "filename": uploaded.name,
                "class": top_class,
                "confidence": top_conf,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.session_state.last_upload_id = upload_id

        st.markdown("<div class='section-label' style='margin-top:2rem'>🩺 Diagnosis Result</div>",
                    unsafe_allow_html=True)
        res_left, res_right = st.columns(2, gap="large")

        with res_left:
            st.markdown(f"""
            <div class="card" style="border-color:{color}55;background:linear-gradient(135deg,{color_soft},var(--card))">
              <span class="result-badge" style="background:{color}22;color:{color};border:1px solid {color}55">{top_class}</span>
              <div class="result-title">{info.get('full_name','')}</div>
              <div class="result-sub">Retinal disease detected</div>
              <div class="result-desc">{info.get('description','')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-label' style='margin-top:1.3rem'>📊 Confidence Breakdown</div>",
                        unsafe_allow_html=True)
            for r in results:
                c_key = DISEASE_INFO.get(r["class"], {}).get("color", "teal")
                confidence_bar(r["class"], r["confidence"], VAR[c_key])

        with res_right:
            if info.get("symptoms"):
                st.markdown("<div class='section-label'>🩻 Common Symptoms</div>", unsafe_allow_html=True)
                items = "".join(f"<li>{s}</li>" for s in info["symptoms"])
                st.markdown(f"<ul class='symptom-list'>{items}</ul>", unsafe_allow_html=True)

            st.markdown("<div class='section-label' style='margin-top:1rem'>💊 Treatment Options</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='treat-box' style='border-color:var(--orange);color:var(--orange);"
                f"background:var(--orange-soft)'>{info.get('treatment','Consult a specialist.')}</div>",
                unsafe_allow_html=True)

            st.markdown("<div class='section-label' style='margin-top:1rem'>🚨 Urgency Level</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='urgency-box' style='border-color:{color};color:{color};"
                f"background:{color_soft}'>{info.get('urgency','')}</div>",
                unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-number" style="color:{color}">{top_conf*100:.1f}%</div>
              <div class="stat-label">TOP PREDICTION CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)

        if top_conf < 0.60:
            st.warning(
                "⚠️ The model's confidence is **below 60 %**. "
                "The image quality may be low, or this could be an atypical case. "
                "Please consult an ophthalmologist for a definitive diagnosis."
            )

    # Disease reference guide
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>📚 Disease Reference Guide</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (code, info) in enumerate(DISEASE_INFO.items()):
        c = VAR[info["color"]]
        with cols[i]:
            st.markdown(f"""
            <div class="ref-card" style="border-color:{c}33">
              <div class="ref-code" style="color:{c}">{code}</div>
              <div class="ref-name">{info['full_name']}</div>
              <div class="ref-desc">{info['description'][:150]}…</div>
              <div class="ref-urgency" style="color:{c}">{info['urgency_short']}</div>
            </div>
            """, unsafe_allow_html=True)


def my_reports_page():
    render_header("My Reports", "📋")
    st.markdown("<div class='section-label'>📋 Scan History (this session)</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            "<div class='preview-empty' style='padding:3rem'>"
            "No scans yet. Head to <b>Scan &amp; Predict</b> to analyse your first OCT image."
            "</div>", unsafe_allow_html=True)
        return

    for rec in st.session_state.history:
        info = DISEASE_INFO.get(rec["class"], {})
        c = VAR[info.get("color", "teal")]
        st.markdown(f"""
        <div class="report-row" style="border-left:3px solid {c}">
          <div>
            <span class="report-class" style="color:{c}">{rec['class']}</span>
            <span class="report-file">{rec['filename']}</span>
          </div>
          <div class="report-meta">
            <span>{rec['confidence']*100:.1f}% confidence</span>
            <span>{rec['time']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.last_upload_id = None
        st.rerun()


def disease_guide_page():
    render_header("Disease Guide", "📖")
    st.markdown("<div class='section-label'>📖 Retinal Disease Reference</div>", unsafe_allow_html=True)

    for code, info in DISEASE_INFO.items():
        c = VAR[info["color"]]
        symptoms_html = ""
        if info.get("symptoms"):
            items = "".join(f"<li>{s}</li>" for s in info["symptoms"])
            symptoms_html = f"<b style='color:var(--text)'>Symptoms</b><ul class='symptom-list'>{items}</ul>"

        st.markdown(f"""
        <div class="card" style="border-color:{c}44; margin-bottom:1.1rem">
          <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.7rem">
            <span class="result-badge" style="background:{c}22;color:{c};border:1px solid {c}55">{code}</span>
            <span style="color:var(--text);font-weight:800;font-size:1.1rem">{info['full_name']}</span>
          </div>
          <p class="result-desc" style="margin-bottom:1rem">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        cgrid1, cgrid2 = st.columns(2)
        with cgrid1:
            if symptoms_html:
                st.markdown(symptoms_html, unsafe_allow_html=True)
        with cgrid2:
            st.markdown(f"<b style='color:var(--text)'>Treatment</b>"
                        f"<p style='color:var(--text-secondary);font-size:.88rem'>{info.get('treatment','')}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"<span style='color:{c};font-weight:700'>{info.get('urgency','')}</span>",
                        unsafe_allow_html=True)
        st.markdown("<hr class='divider' style='margin:1.4rem 0'>", unsafe_allow_html=True)


def how_it_works_page():
    render_header("How It Works", "⚙️")
    st.markdown("<div class='section-label'>⚙️ Prediction Pipeline</div>", unsafe_allow_html=True)

    steps = [
        ("1", "📤", "Upload", "Upload a retinal OCT scan in JPG, PNG, BMP, or TIFF format."),
        ("2", "🧮", "Preprocess", "The image is resized to 224×224 and pixel values are normalised to the [0, 1] range."),
        ("3", "🧠", "Inference", f"A {MODEL_BADGE} convolutional network, fine-tuned on the Kaggle OCT 2017 dataset, "
                                   "predicts probabilities across four classes: CNV, DME, Drusen, and Normal."),
        ("4", "🩺", "Result", "The highest-probability class is shown alongside symptoms, treatment guidance, "
                                "and an urgency level to help you decide next steps."),
    ]
    for num, icon, title, desc in steps:
        st.markdown(f"""
        <div class="card" style="display:flex;gap:1.1rem;align-items:flex-start;margin-bottom:1rem">
          <div class="step-num">{num}</div>
          <div>
            <div style="font-weight:800;color:var(--text);font-size:1.02rem">{icon}&nbsp;&nbsp;{title}</div>
            <div style="color:var(--text-secondary);font-size:.9rem;margin-top:.35rem;line-height:1.55">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <b style="color:var(--text)">⚠️ Disclaimer</b>
      <p style="color:var(--text-secondary);font-size:.88rem;margin-top:.5rem;line-height:1.6">
        This tool is for educational purposes only and does NOT replace professional
        medical diagnosis. Always consult a qualified ophthalmologist for any concerns
        about your eye health.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
inject_css()
render_sidebar()

PAGES = {
    "scan": scan_predict_page,
    "reports": my_reports_page,
    "guide": disease_guide_page,
    "how": how_it_works_page,
}
PAGES.get(st.session_state.page, scan_predict_page)()
