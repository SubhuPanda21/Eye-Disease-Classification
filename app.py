# =============================================================================
#  OcuScan AI — Advanced Retinal Disease Classification System
#  EfficientNetB4 + EfficientNetV2S + ConvNeXt-Small Ensemble
#  For Research / Clinical Decision Support Use Only
# =============================================================================

import os, sys, io, json, time, datetime, base64, hashlib, pathlib
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import plotly.graph_objects as go
import plotly.express as px

# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not found — running in demo/mock mode.")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage,
                                     HRFlowable)
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR        = os.environ.get("OCUSCAN_MODEL_DIR", "/kaggle/working/")
IMG_SIZE        = 256        # eye_model.keras input: 256×256×3
TTA_STEPS       = 5          # Test-Time Augmentation passes

APP_VERSION     = "2.1.0"
DISCLAIMER      = (
    "⚕️ **Medical Disclaimer:** OcuScan AI is intended for research and clinical "
    "decision *support* only. It is NOT a substitute for professional ophthalmic "
    "examination. All findings must be reviewed and confirmed by a licensed "
    "ophthalmologist before any clinical action is taken."
)

MODEL_FILE      = "eye_model.keras"

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
]

CLASS_INFO = {
    "Normal": {
        "severity": "None", "color": "#4ade80", "icd": "Z01.01",
        "description": "No pathological findings detected in the retinal image.",
        "recommendations": [
            "Routine annual screening recommended.",
            "Maintain healthy blood pressure and blood glucose.",
            "Use UV-protective eyewear outdoors.",
        ],
        "urgency": "Routine",
    },
    "Diabetic Retinopathy": {
        "severity": "High", "color": "#f87171", "icd": "E11.319",
        "description": "Microvascular retinal damage secondary to chronic hyperglycemia. Signs may include microaneurysms, hemorrhages, exudates, or neovascularization.",
        "recommendations": [
            "Urgent referral to retinal specialist.",
            "Strict glycemic and blood pressure control (HbA1c < 7%).",
            "Anti-VEGF therapy or laser photocoagulation may be indicated.",
            "Follow-up in 1–3 months depending on severity.",
        ],
        "urgency": "Urgent",
    },
    "Glaucoma": {
        "severity": "High", "color": "#fb923c", "icd": "H40.10",
        "description": "Progressive optic neuropathy with characteristic optic disc cupping and visual field loss, often associated with elevated intraocular pressure.",
        "recommendations": [
            "Refer to glaucoma specialist within 2 weeks.",
            "IOP measurement and visual field testing required.",
            "Consider tonometry and OCT of optic nerve head.",
            "Topical IOP-lowering agents may be initiated.",
        ],
        "urgency": "Semi-Urgent",
    },
    "Cataract": {
        "severity": "Moderate", "color": "#facc15", "icd": "H26.9",
        "description": "Opacification of the crystalline lens causing progressive visual impairment. May appear as nuclear, cortical, or posterior subcapsular opacity.",
        "recommendations": [
            "Ophthalmology referral for surgical evaluation.",
            "Phacoemulsification with IOL implantation when visually significant.",
            "Monitor visual acuity and functional impact.",
        ],
        "urgency": "Elective",
    },
}

URGENCY_COLORS = {
    "Urgent": "#f87171",
    "Semi-Urgent": "#fb923c",
    "Elective": "#facc15",
    "Routine": "#4ade80",
    "Review": "#94a3b8",
    "None": "#4ade80",
}

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OcuScan AI — Retinal Disease Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #020818 !important;
    color: #e2e8f0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* Main content */
.main .block-container { padding-top: 2rem; max-width: 1400px; }

/* Cards */
.ocuscan-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}
.result-card {
    background: rgba(15,23,42,0.9);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: rgba(0,255,180,0.08) !important;
    border: 1px solid rgba(0,255,180,0.3) !important;
    color: #4fffcc !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: rgba(0,255,180,0.15) !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.2) !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
.stFileUploader {
    background: rgba(15,23,42,0.6) !important;
    border: 2px dashed rgba(0,255,180,0.2) !important;
    border-radius: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(15,23,42,0.6) !important; border-radius: 10px !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
.stTabs [aria-selected="true"] { color: #4fffcc !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #00ffb4, #00d4ff) !important; border-radius: 4px; }

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; background: rgba(15,23,42,0.8) !important; }

/* Headers */
h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

/* Scanline overlay */
.scanline-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,180,0.012) 2px, rgba(0,255,180,0.012) 4px
    );
    pointer-events: none; z-index: 9999;
}

/* Grid background */
body::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,255,180,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,180,0.025) 1px, transparent 1px);
    background-size: 50px 50px;
    z-index: -1;
}

/* Badge */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 700;
}

/* Mono text */
.mono { font-family: 'Space Mono', monospace !important; font-size: 11px !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_models():
    """Return dict of {model_name: (exists, path, size_mb)}."""
    path   = os.path.join(SAVE_DIR, MODEL_FILE)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) / 1e6 if exists else 0
    return {"eye_model": (exists, path, size)}


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load eye_model.keras; return dict with single entry or empty dict on failure."""
    if not TF_AVAILABLE:
        return {}
    path = os.path.join(SAVE_DIR, MODEL_FILE)
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path, compile=False)
            return {"eye_model": model}
        except Exception as e:
            st.sidebar.error(f"❌ eye_model: {e}")
    else:
        st.sidebar.warning(f"⚠️ eye_model.keras not found at {path}")
    return {}


def preprocess_image(pil_img, target_size=IMG_SIZE):
    """Resize + normalize to [0,1] float32 array."""
    img = pil_img.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def tta_augment(arr):
    """Yield TTA_STEPS augmented versions of a (1,H,W,3) array."""
    yield arr
    yield arr[:, :, ::-1, :]                          # horizontal flip
    yield arr[:, ::-1, :, :]                          # vertical flip
    yield np.rot90(arr, k=1, axes=(1, 2))             # 90°
    yield np.rot90(arr, k=3, axes=(1, 2))             # 270°


def run_model(model, arr):
    """Run TTA inference; return mean probability vector."""
    preds = []
    for aug in tta_augment(arr):
        resized = tf.image.resize(aug, [IMG_SIZE, IMG_SIZE]).numpy()
        preds.append(model.predict(resized, verbose=0)[0])
    return np.mean(preds, axis=0)


def mock_inference(n_classes=4):
    """Return fake softmax scores for demo mode."""
    raw = np.random.dirichlet(np.ones(n_classes) * 0.4)
    # Boost a random class to simulate a real prediction
    winner = np.random.randint(n_classes)
    raw[winner] += 0.5
    return raw / raw.sum()


def ensemble_predict(models, pil_img):
    """Run single model and return softmax scores."""
    arr_base = preprocess_image(pil_img, IMG_SIZE)
    all_preds = {}

    if not models:
        # Demo mode: mock the model
        all_preds["eye_model"] = mock_inference(len(CLASSES))
    else:
        for name, model in models.items():
            arr = preprocess_image(pil_img, IMG_SIZE)
            all_preds[name] = run_model(model, arr)

    ensemble_scores = np.mean(list(all_preds.values()), axis=0)
    return ensemble_scores, all_preds


def image_quality_check(pil_img):
    """Heuristic image quality assessment."""
    img = pil_img.convert("RGB")
    arr = np.array(img)
    brightness = arr.mean() / 255
    contrast   = arr.std() / 255
    blur_score = np.array(img.filter(ImageFilter.FIND_EDGES)).mean()
    issues = []
    if brightness < 0.15: issues.append("Image too dark")
    if brightness > 0.85: issues.append("Image overexposed")
    if contrast < 0.05:   issues.append("Low contrast")
    if blur_score < 8:    issues.append("Possible motion blur / out of focus")
    quality = "Good" if not issues else ("Acceptable" if len(issues) == 1 else "Poor")
    return quality, issues, round(brightness, 2), round(contrast, 2), round(blur_score, 1)


def generate_pdf_report(patient_info, scan_info, scores, per_model, image_bytes):
    """Generate a clinical-style PDF report; returns bytes."""
    if not REPORTLAB_AVAILABLE:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontName="Helvetica-Bold", fontSize=20,
                                  textColor=colors.HexColor("#0f172a"), spaceAfter=4)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontName="Helvetica", fontSize=9,
                                  textColor=colors.HexColor("#64748b"))
    head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                  fontName="Helvetica-Bold", fontSize=11,
                                  textColor=colors.HexColor("#0f172a"), spaceBefore=12, spaceAfter=4)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontName="Helvetica", fontSize=9,
                                  textColor=colors.HexColor("#1e293b"), leading=14)
    warn_style  = ParagraphStyle("warn", parent=styles["Normal"],
                                  fontName="Helvetica-Oblique", fontSize=8,
                                  textColor=colors.HexColor("#dc2626"), leading=12)

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("OcuScan AI — Clinical Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')} | "
        f"App v{APP_VERSION} | For Research / Decision Support Use Only",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=10))

    # ── Patient info ──────────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", head_style))
    pt_data = [
        ["Name", patient_info.get("name", "—"),      "Patient ID", patient_info.get("id", "—")],
        ["Age",  patient_info.get("age", "—"),        "Sex",        patient_info.get("sex", "—")],
        ["Eye",  scan_info.get("eye", "—"),           "Scan Type",  scan_info.get("scan_type", "—")],
        ["Referring Physician", patient_info.get("referring_physician", "—"), "Date", scan_info.get("date", datetime.date.today().isoformat())],
    ]
    t = Table(pt_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 6*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0),(2,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0),(0,-1),  colors.HexColor("#64748b")),
        ("TEXTCOLOR",   (2,0),(2,-1),  colors.HexColor("#64748b")),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND",  (0,0),(0,-1),  colors.HexColor("#f8fafc")),
        ("BACKGROUND",  (2,0),(2,-1),  colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Primary result ────────────────────────────────────────────────────────
    story.append(Paragraph("AI Classification Result", head_style))
    pred_idx  = int(np.argmax(scores))
    pred_name = CLASSES[pred_idx]
    info      = CLASS_INFO[pred_name]
    conf      = float(scores[pred_idx]) * 100

    result_data = [
        ["Primary Diagnosis", pred_name],
        ["Confidence",        f"{conf:.1f}%"],
        ["ICD-10 Code",       info["icd"]],
        ["Urgency",           info["urgency"]],
        ["Severity",          info["severity"]],
    ]
    rt = Table(result_data, colWidths=[5*cm, 14*cm])
    rt.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0),(0,-1),  colors.HexColor("#64748b")),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(rt)
    story.append(Spacer(1, 8))
    story.append(Paragraph(info["description"], body_style))
    story.append(Spacer(1, 10))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Recommendations", head_style))
    for i, rec in enumerate(info["recommendations"], 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
    story.append(Spacer(1, 10))

    # ── Ensemble scores ───────────────────────────────────────────────────────
    story.append(Paragraph("Ensemble Model Scores", head_style))
    score_rows = [["Class", "Probability", "Bar"]]
    for i, (cls, sc) in enumerate(zip(CLASSES, scores)):
        bar = "█" * int(sc * 30) + "░" * (30 - int(sc * 30))
        score_rows.append([cls, f"{sc*100:.1f}%", bar])
    st2 = Table(score_rows, colWidths=[6*cm, 3*cm, 10*cm])
    st2.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("FONTNAME",    (2,1),(-1,-1), "Courier"),
        ("FONTSIZE",    (2,1),(-1,-1), 7),
        ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",   (0,0),(-1,0),  colors.white),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(st2)
    story.append(Spacer(1, 12))

    # ── Per-model scores ──────────────────────────────────────────────────────
    story.append(Paragraph("Per-Model Predictions", head_style))
    pm_header = ["Class"] + list(per_model.keys())
    pm_rows   = [pm_header]
    for i, cls in enumerate(CLASSES):
        row = [cls] + [f"{per_model[m][i]*100:.1f}%" for m in per_model]
        pm_rows.append(row)
    ncols = len(pm_header)
    col_w = [6*cm] + [4.5*cm] * (ncols - 1)
    pmt = Table(pm_rows, colWidths=col_w)
    pmt.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",   (0,0),(-1,0),  colors.white),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(pmt)
    story.append(Spacer(1, 16))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for research and clinical decision "
        "support purposes only. It is NOT a definitive medical diagnosis. All findings must be "
        "reviewed and confirmed by a qualified ophthalmologist. OcuScan AI and its developers "
        "assume no liability for clinical decisions made on the basis of this report.",
        warn_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 8px;">
        <div style="font-size:40px; margin-bottom:8px;">👁️</div>
        <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:800;
                    background:linear-gradient(135deg,#fff 30%,#4fffcc 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            OcuScan AI
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:10px; color:#475569;
                    letter-spacing:0.1em; margin-top:4px;">
            RETINAL ANALYSIS SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown("**🔬 Model Status**")
    model_status = check_models()
    all_found = all(v[0] for v in model_status.values())
    for name, (exists, path, size) in model_status.items():
        icon = "✅" if exists else "❌"
        col1, col2 = st.columns([3,1])
        col1.markdown(f"<span class='mono'>{icon} {name}</span>", unsafe_allow_html=True)
        col2.markdown(f"<span class='mono' style='color:#64748b'>{size:.0f}MB</span>", unsafe_allow_html=True)

    if not all_found:
        st.markdown(f"""
        <div style='background:rgba(248,113,113,0.1); border:1px solid rgba(248,113,113,0.3);
                    border-radius:8px; padding:10px; margin-top:8px; font-size:11px; color:#fca5a5;
                    font-family:"Space Mono",monospace;'>
        ⚠️ Missing models detected.<br>
        Place .keras files in:<br>
        <code>{SAVE_DIR}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("All models loaded ✓")

    st.divider()

    # ── Analysis settings ─────────────────────────────────────────────────────
    st.markdown("**⚙️ Analysis Settings**")
    use_tta       = st.toggle("Test-Time Augmentation (TTA)", value=True)
    show_per_model = st.toggle("Show per-model scores", value=True)
    confidence_threshold = st.slider("Confidence threshold for alert", 0.5, 0.99, 0.85, 0.01)

    st.divider()

    # ── Model directory ───────────────────────────────────────────────────────
    st.markdown("**📂 Model Directory**")
    custom_dir = st.text_input("Override path", value=SAVE_DIR, label_visibility="collapsed")
    if custom_dir != SAVE_DIR:
        SAVE_DIR = custom_dir
        st.cache_resource.clear()

    st.divider()
    st.markdown(f"""
    <div style='font-family:"Space Mono",monospace; font-size:10px; color:#334155; line-height:1.8;'>
    v{APP_VERSION} · eye_model.keras<br>
    Input: 256×256×3 · 4-class<br>
    TensorFlow {'✓' if TF_AVAILABLE else '✗'} · ReportLab {'✓' if REPORTLAB_AVAILABLE else '✗'}<br><br>
    For Research Use Only
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:32px;">
    <div style="font-family:'Space Mono',monospace; font-size:11px; color:#4fffcc;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;
                display:inline-block; padding:4px 14px; border:1px solid rgba(0,255,180,0.2);
                border-radius:20px; background:rgba(0,255,180,0.05);">
        👁 OcuScan AI · Clinical Decision Support
    </div>
    <h1 style="font-family:'Syne',sans-serif; font-size:clamp(28px,4vw,52px);
               font-weight:800; letter-spacing:-0.03em; line-height:1.05;
               background:linear-gradient(135deg,#ffffff 30%,#4fffcc 100%);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Retinal Disease<br>Classification
    </h1>
    <p style="color:#64748b; font-size:14px; margin-top:8px; max-width:600px;">
        Single-model inference — eye_model.keras &nbsp;·&nbsp;
        4-class fundus pathology screening
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(DISCLAIMER)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_scan, tab_history, tab_about = st.tabs(["🔬 New Scan", "📋 Session History", "ℹ️ About"])

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load models once ──────────────────────────────────────────────────────────
models = load_all_models()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: NEW SCAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:

    # ── Patient Information ───────────────────────────────────────────────────
    with st.expander("👤 Patient Information (optional — for report generation)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        pt_name   = c1.text_input("Patient Name",  placeholder="Full name")
        pt_id     = c2.text_input("Patient ID",    placeholder="MRN / Hospital ID")
        pt_age    = c3.text_input("Age",            placeholder="e.g. 54")
        pt_sex    = c4.selectbox("Sex", ["—", "Male", "Female", "Other"])

        c5, c6, c7, c8 = st.columns(4)
        pt_ref    = c5.text_input("Referring Physician", placeholder="Dr. Name")
        pt_eye    = c6.selectbox("Eye", ["—", "Right (OD)", "Left (OS)", "Both (OU)"])
        pt_scan   = c7.selectbox("Scan Type", ["Fundus Photography", "Slit-lamp", "OCT", "Other"])
        pt_notes  = c8.text_area("Clinical Notes", placeholder="Symptoms, history…", height=72)

    patient_info = {"name": pt_name, "id": pt_id, "age": pt_age, "sex": pt_sex,
                    "referring_physician": pt_ref}
    scan_info    = {"eye": pt_eye, "scan_type": pt_scan, "notes": pt_notes,
                    "date": datetime.date.today().isoformat()}

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload Retinal Scan")
        st.markdown("<p style='font-size:12px; color:#64748b;'>Supports JPG, JPEG, PNG — fundus photography, slit-lamp & OCT images</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

        if uploaded:
            image_bytes = uploaded.read()
            pil_img     = Image.open(io.BytesIO(image_bytes))
            quality, issues, brightness, contrast, blur = image_quality_check(pil_img)

            # Quality badge
            qcolor = {"Good": "#4ade80", "Acceptable": "#facc15", "Poor": "#f87171"}.get(quality, "#94a3b8")
            st.markdown(f"""
            <div style="margin:12px 0; padding:12px 16px;
                        background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-family:'Space Mono',monospace; font-size:11px; color:#94a3b8;">IMAGE QUALITY</span>
                    <span class="badge" style="background:{qcolor}22; color:{qcolor}; border:1px solid {qcolor}44;">
                        {quality}
                    </span>
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:10px; color:#64748b; line-height:1.8;">
                    Brightness: {brightness:.2f} &nbsp;·&nbsp; Contrast: {contrast:.2f} &nbsp;·&nbsp; Edge score: {blur:.1f}
                </div>
                {"".join(f'<div style="font-size:11px; color:#f87171; margin-top:4px;">⚠ {i}</div>' for i in issues)}
            </div>
            """, unsafe_allow_html=True)

            run_btn = st.button("🔬 Run Ensemble Analysis", use_container_width=True)

    with col_preview:
        if uploaded:
            st.markdown("### 🖼️ Scan Preview")
            disp_img = pil_img.copy()
            disp_img.thumbnail((500, 500), Image.LANCZOS)

            # Preprocessing toggle
            enhance = st.toggle("CLAHE-style contrast enhancement", value=False)
            if enhance:
                enhancer = ImageEnhance.Contrast(disp_img)
                disp_img = enhancer.enhance(2.0)
                enhancer2 = ImageEnhance.Sharpness(disp_img)
                disp_img  = enhancer2.enhance(1.5)

            st.image(disp_img, use_container_width=True,
                     caption=f"{uploaded.name} · {pil_img.size[0]}×{pil_img.size[1]}px")

    # ── Run inference ─────────────────────────────────────────────────────────
    if uploaded and run_btn:
        st.divider()
        st.markdown("### ⚙️ Ensemble Inference Pipeline")

        progress_bar = st.progress(0, text="Initialising…")
        status_placeholder = st.empty()

        def update_status(state):
            icons = {"waiting": "⬜", "running": "🔄", "done": "✅", "error": "❌"}
            colors_map = {"waiting": "#475569", "running": "#4fffcc", "done": "#4ade80", "error": "#f87171"}
            status_placeholder.markdown(
                f"<div style='text-align:center; padding:12px; background:rgba(15,23,42,0.8); "
                f"border:1px solid rgba(255,255,255,0.06); border-radius:12px; max-width:300px; margin:0 auto;'>"
                f"<div style='font-size:20px'>{icons[state]}</div>"
                f"<div style='font-family:\"Space Mono\",monospace; font-size:10px; "
                f"color:{colors_map[state]}; margin-top:4px; letter-spacing:0.05em;'>eye_model.keras</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        update_status("waiting")

        all_preds_per_model = {}
        total_steps = 3

        progress_bar.progress(1 / total_steps, text="Preprocessing image…")
        time.sleep(0.3)

        update_status("running")
        progress_bar.progress(2 / total_steps, text="Running eye_model…")
        try:
            if models and "eye_model" in models:
                arr = preprocess_image(pil_img, IMG_SIZE)
                if use_tta:
                    pred = run_model(models["eye_model"], arr)
                else:
                    pred = models["eye_model"].predict(preprocess_image(pil_img, IMG_SIZE), verbose=0)[0]
            else:
                time.sleep(0.5)
                pred = mock_inference(len(CLASSES))
            all_preds_per_model["eye_model"] = pred
            update_status("done")
        except Exception as e:
            update_status("error")
            st.error(f"eye_model failed: {e}")
            all_preds_per_model["eye_model"] = mock_inference(len(CLASSES))

        progress_bar.progress(total_steps / total_steps, text="Analysis complete ✓")
        ensemble_scores = all_preds_per_model["eye_model"]
        time.sleep(0.2)

        # ── Results ───────────────────────────────────────────────────────────
        pred_idx  = int(np.argmax(ensemble_scores))
        pred_name = CLASSES[pred_idx]
        info      = CLASS_INFO[pred_name]
        conf      = float(ensemble_scores[pred_idx])
        urg_color = URGENCY_COLORS.get(info["urgency"], "#94a3b8")

        st.divider()
        st.markdown("### 📊 Analysis Results")

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Primary Diagnosis", pred_name)
        m2.metric("Confidence", f"{conf*100:.1f}%")
        m3.metric("Urgency", info["urgency"])
        m4.metric("ICD-10", info["icd"])

        st.divider()
        res_col, chart_col = st.columns([1, 1], gap="large")

        # ── Left: Clinical summary ────────────────────────────────────────────
        with res_col:
            pcolor = info["color"]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{pcolor}11,{pcolor}06);
                        border:1px solid {pcolor}33; border-radius:16px;
                        padding:20px; margin-bottom:16px;
                        box-shadow:0 0 24px {pcolor}15;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                    <div>
                        <div style="font-family:'Space Mono',monospace; font-size:10px;
                                    color:#64748b; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;">
                            AI CLASSIFICATION
                        </div>
                        <div style="font-size:22px; font-weight:800; color:{pcolor}; font-family:'Syne',sans-serif;">
                            {pred_name}
                        </div>
                        <div style="font-size:12px; color:#64748b; margin-top:4px;">
                            ICD-10: {info["icd"]} · Severity: {info["severity"]}
                        </div>
                    </div>
                    <div style="text-align:center; min-width:70px; padding:12px;
                                background:{pcolor}15; border:1px solid {pcolor}33;
                                border-radius:12px;">
                        <div style="font-family:'Space Mono',monospace; font-size:20px;
                                    font-weight:700; color:{pcolor};">
                            {conf*100:.0f}%
                        </div>
                        <div style="font-size:9px; color:#64748b; margin-top:2px; letter-spacing:0.05em;">
                            CONFIDENCE
                        </div>
                    </div>
                </div>
                <div style="padding:12px; background:rgba(0,0,0,0.2); border-radius:8px;
                            font-size:13px; color:#94a3b8; line-height:1.7; margin-bottom:12px;">
                    {info["description"]}
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                    <span style="font-family:'Space Mono',monospace; font-size:10px; color:#64748b;">URGENCY:</span>
                    <span class="badge" style="background:{urg_color}22; color:{urg_color}; border:1px solid {urg_color}44;">
                        {info["urgency"]}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence threshold alert
            if conf < confidence_threshold:
                st.warning(f"⚠️ Confidence ({conf*100:.1f}%) is below your threshold ({confidence_threshold*100:.0f}%). Manual review strongly recommended.")

            # Recommendations
            st.markdown("**🩺 Clinical Recommendations**")
            for i, rec in enumerate(info["recommendations"], 1):
                st.markdown(f"""
                <div style="display:flex; gap:12px; align-items:flex-start; padding:10px 12px;
                            background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.05);
                            border-radius:8px; margin-bottom:6px; font-size:13px; color:#94a3b8;">
                    <span style="color:{pcolor}; font-weight:700; font-family:'Space Mono',monospace; min-width:16px;">{i}</span>
                    {rec}
                </div>
                """, unsafe_allow_html=True)

            if pt_notes:
                st.markdown("**📝 Clinical Notes**")
                st.info(pt_notes)

        # ── Right: Charts ─────────────────────────────────────────────────────
        with chart_col:
            # Radial / bar chart of ensemble scores
            sorted_idx = np.argsort(ensemble_scores)[::-1]
            sorted_names  = [CLASSES[i] for i in sorted_idx]
            sorted_scores = [float(ensemble_scores[i]) for i in sorted_idx]
            sorted_colors = [CLASS_INFO[CLASSES[i]]["color"] for i in sorted_idx]

            fig_bar = go.Figure(go.Bar(
                x=sorted_scores,
                y=sorted_names,
                orientation="h",
                marker=dict(
                    color=sorted_colors,
                    opacity=[1.0 if CLASSES[i] == pred_name else 0.35 for i in sorted_idx],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{s*100:.1f}%" for s in sorted_scores],
                textposition="outside",
                textfont=dict(family="Space Mono", size=10, color="#94a3b8"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Syne", color="#94a3b8", size=11),
                margin=dict(l=0, r=60, t=30, b=20),
                xaxis=dict(showgrid=False, zeroline=False, range=[0, 1.1],
                           tickformat=".0%", tickfont=dict(family="Space Mono", size=9),
                           color="#475569"),
                yaxis=dict(showgrid=False, color="#94a3b8",
                           tickfont=dict(family="Syne", size=11)),
                title=dict(text="Ensemble Probability Distribution", font=dict(size=13, color="#e2e8f0")),
                height=360,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # Per-model comparison (single model radar)
            if show_per_model and all_preds_per_model:
                fig_pm = go.Figure()
                model_colors = {"eye_model": "#4fffcc"}
                for mname, mpred in all_preds_per_model.items():
                    fig_pm.add_trace(go.Scatterpolar(
                        r=[float(v) for v in mpred],
                        theta=CLASSES,
                        fill="toself",
                        name=mname,
                        line=dict(color=model_colors.get(mname, "#4fffcc"), width=2),
                        fillcolor="rgba(79,255,204,0.08)",
                        opacity=0.9,
                    ))
                fig_pm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.05)",
                                        tickfont=dict(family="Space Mono", size=8, color="#475569")),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                                         tickfont=dict(family="Syne", size=9, color="#94a3b8")),
                    ),
                    legend=dict(font=dict(family="Space Mono", size=9, color="#94a3b8"),
                                bgcolor="rgba(0,0,0,0)"),
                    title=dict(text="Per-Model Radar", font=dict(size=13, color="#e2e8f0")),
                    margin=dict(l=20, r=20, t=40, b=20), height=320,
                    font=dict(family="Syne", color="#94a3b8"),
                )
                st.plotly_chart(fig_pm, use_container_width=True, config={"displayModeBar": False})

        # ── Per-model score table ─────────────────────────────────────────────
        if show_per_model and all_preds_per_model:
            st.divider()
            st.markdown("### 🔍 Detailed Per-Model Scores")
            import pandas as pd
            df_rows = {}
            for cls_idx, cls in enumerate(CLASSES):
                df_rows[cls] = {m: f"{all_preds_per_model[m][cls_idx]*100:.1f}%" for m in all_preds_per_model}
                df_rows[cls]["Ensemble"] = f"{ensemble_scores[cls_idx]*100:.1f}%"
            df = pd.DataFrame(df_rows).T
            df.index.name = "Class"
            st.dataframe(df.style.applymap(lambda v: "color: #4fffcc; font-weight: bold;"
                                             if v == f"{conf*100:.1f}%" else "color: #94a3b8"),
                         use_container_width=True)

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Export Report")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        # JSON export
        report_json = {
            "report_version": APP_VERSION,
            "generated_at": datetime.datetime.now().isoformat(),
            "patient": patient_info,
            "scan": scan_info,
            "result": {
                "prediction": pred_name,
                "confidence": float(conf),
                "icd10": info["icd"],
                "urgency": info["urgency"],
                "severity": info["severity"],
            },
            "ensemble_scores": {CLASSES[i]: float(ensemble_scores[i]) for i in range(len(CLASSES))},
            "model_scores": {"eye_model": {CLASSES[i]: float(v) for i, v in enumerate(all_preds_per_model["eye_model"])}},
            "image_quality": {"quality": quality, "issues": issues,
                               "brightness": brightness, "contrast": contrast, "blur_score": blur},
            "disclaimer": "For research and clinical decision support only."
        }
        exp_col1.download_button("⬇ Download JSON Report",
                                  data=json.dumps(report_json, indent=2),
                                  file_name=f"ocuscan_{pt_id or 'unknown'}_{datetime.date.today()}.json",
                                  mime="application/json", use_container_width=True)

        # PDF export
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_report(patient_info, scan_info, ensemble_scores,
                                             all_preds_per_model, image_bytes)
            if pdf_bytes:
                exp_col2.download_button("⬇ Download PDF Report",
                                          data=pdf_bytes,
                                          file_name=f"ocuscan_{pt_id or 'unknown'}_{datetime.date.today()}.pdf",
                                          mime="application/pdf", use_container_width=True)
        else:
            exp_col2.markdown("<span style='font-size:11px; color:#475569; font-family:\"Space Mono\",monospace;'>PDF: install reportlab</span>", unsafe_allow_html=True)

        # Image + result composite
        exp_col3.markdown(f"""
        <div style="padding:10px; background:rgba(15,23,42,0.8);
                    border:1px solid rgba(255,255,255,0.06); border-radius:10px;
                    font-family:'Space Mono',monospace; font-size:10px; color:#64748b; text-align:center;">
            Case ID: {hashlib.md5((pt_id or uploaded.name).encode()).hexdigest()[:8].upper()}
        </div>
        """, unsafe_allow_html=True)

        # ── Save to history ───────────────────────────────────────────────────
        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "file": uploaded.name,
            "patient": pt_name or "Anonymous",
            "prediction": pred_name,
            "confidence": f"{conf*100:.1f}%",
            "urgency": info["urgency"],
            "icd10": info["icd"],
        })


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 📋 Session Scan History")
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#334155;
                    font-family:'Space Mono',monospace; font-size:12px;">
            No scans analysed this session yet.
        </div>
        """, unsafe_allow_html=True)
    else:
        import pandas as pd
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        # Summary stats
        if len(st.session_state.history) > 1:
            st.divider()
            st.markdown("### 📈 Session Summary")
            from collections import Counter
            pred_counts = Counter(h["prediction"] for h in st.session_state.history)
            fig_hist = go.Figure(go.Bar(
                x=list(pred_counts.keys()),
                y=list(pred_counts.values()),
                marker_color=[CLASS_INFO.get(k, {}).get("color", "#94a3b8") for k in pred_counts],
                opacity=0.85,
            ))
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Syne", color="#94a3b8"),
                xaxis=dict(tickfont=dict(size=10), color="#64748b", showgrid=False),
                yaxis=dict(tickfont=dict(size=10), color="#64748b", gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=20, t=20, b=60),
                height=280,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### ℹ️ About OcuScan AI")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        **OcuScan AI** is a deep learning–based system for automated screening
        of retinal pathologies from fundus photography and related imaging modalities.

        **Model Architecture**
        - eye_model.keras (Custom CNN)
        - Input: 256 × 256 × 3
        - Output: 4 classes

        Predictions are generated from the single model (with optional
        Test-Time Augmentation passes) to produce a robust score.

        **Supported Conditions**
        """)
        for cls, info in CLASS_INFO.items():
            c = info["color"]
            st.markdown(f"""
            <div style="display:flex; gap:12px; align-items:center; margin-bottom:6px;
                        padding:8px 12px; background:rgba(15,23,42,0.8);
                        border:1px solid rgba(255,255,255,0.05); border-radius:8px;">
                <span style="width:10px; height:10px; border-radius:50%;
                              background:{c}; flex-shrink:0; display:inline-block;
                              box-shadow:0 0 6px {c}66;"></span>
                <span style="font-size:13px; color:#e2e8f0; font-weight:600;">{cls}</span>
                <span style="font-family:'Space Mono',monospace; font-size:10px;
                              color:#475569; margin-left:auto;">{info["icd"]}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Technical Details**")
        tech_items = [
            ("Framework", "TensorFlow / Keras 3.10.0"),
            ("Model", "eye_model.keras"),
            ("Input resolution", "256 × 256 px"),
            ("Output classes", "4"),
            ("TTA passes", str(TTA_STEPS)),
            ("App version", APP_VERSION),
            ("TensorFlow", "Available ✓" if TF_AVAILABLE else "Not found ✗"),
            ("ReportLab PDF", "Available ✓" if REPORTLAB_AVAILABLE else "Not found ✗"),
        ]
        for k, v in tech_items:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 12px;
                        background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px; margin-bottom:4px;">
                <span style="font-family:'Space Mono',monospace; font-size:11px; color:#64748b;">{k}</span>
                <span style="font-family:'Space Mono',monospace; font-size:11px; color:#e2e8f0;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.error("""
**⚕️ Important Notice**

This system is intended for **research and clinical decision support only**.
It is NOT approved as a medical device and must not be used as the sole basis
for clinical diagnosis or treatment decisions.

All AI-generated findings must be reviewed by a **licensed ophthalmologist**
before any clinical action is taken.
        """)