import streamlit as st
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedScan — Medical Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

html, body, [class*="css"] {
    font-family: 'Satoshi', 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1c1b19;
    border-right: 1px solid #2a2928;
}
[data-testid="stSidebar"] * { color: #cdccca !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f0efed !important; }

/* Main background */
.stApp { background: #171614; color: #cdccca; }

/* Cards */
.result-card {
    background: #1c1b19;
    border: 1px solid #2a2928;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.prediction-badge {
    display: inline-block;
    background: #01696f22;
    color: #4f98a3;
    border: 1px solid #4f98a344;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.confidence-big {
    font-size: 3rem;
    font-weight: 700;
    color: #f0efed;
    line-height: 1;
}
.confidence-label {
    font-size: 0.85rem;
    color: #797876;
    margin-top: 0.25rem;
}
.disease-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0efed;
    margin-top: 0.75rem;
}
.disease-desc {
    font-size: 0.9rem;
    color: #797876;
    margin-top: 0.25rem;
    line-height: 1.6;
}
.warning-box {
    background: #56494222;
    border: 1px solid #bb653b44;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #bb653b;
    margin-top: 1rem;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-chip {
    background: #22211f;
    border: 1px solid #2a2928;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    flex: 1;
    text-align: center;
}
.metric-chip .val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0efed;
}
.metric-chip .key {
    font-size: 0.72rem;
    color: #797876;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
/* Upload zone */
[data-testid="stFileUploader"] {
    background: #1c1b19;
    border: 1.5px dashed #393836;
    border-radius: 12px;
}
/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #22211f;
    border: 1px solid #393836;
    border-radius: 8px;
    color: #cdccca;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE CONFIG — add your model path + class labels here
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CONFIG = {
    "🔬 Retinal OCT (Eye)": {
        "description": "Optical Coherence Tomography — retinal layer analysis",
        "model_arch": "convnext",
        "model_path": "/Users/ebbb/Desktop/UTS/S3/DL/Project/convnext_best.pth",
        "num_classes": 4,
        "classes": ["CNV", "DME", "DRUSEN", "NORMAL"],
        "class_info": {
            "CNV":    ("Choroidal Neovascularisation", "Abnormal blood vessel growth beneath the retina. Requires urgent ophthalmology review."),
            "DME":    ("Diabetic Macular Edema",       "Fluid accumulation in the macula due to diabetic retinopathy. Monitor closely."),
            "DRUSEN": ("Drusen Deposits",              "Yellow lipid deposits beneath the retina. Associated with early-stage AMD."),
            "NORMAL": ("No Pathology Detected",        "Retinal layers appear within normal limits. Routine follow-up recommended."),
        },
        "color": "#4f98a3",
    },
    "🧠 Brain Tumour (MRI)": {
        "description": "MRI-based brain tumour classification",
        "model_arch": "vit",
        "model_path": "/home/sagemaker-user/brain-tumour/best_model.pth",
        "num_classes": 4,
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "class_info": {
            "glioma":     ("Glioma",               "A tumour arising from glial cells. Can be low or high grade — prompt neurology referral required."),
            "meningioma": ("Meningioma",            "Typically benign tumour arising from meninges. Often slow-growing; surgical assessment advised."),
            "notumor":    ("No Tumour Detected",    "MRI scan appears within normal limits. Clinical correlation recommended."),
            "pituitary":  ("Pituitary Adenoma",     "Benign tumour of the pituitary gland. May affect hormone levels; endocrinology review needed."),
        },
        "color": "#a86fdf",
    },
    "🫁 Lung Cancer (CT)": {
        "description": "CT scan-based lung pathology classification",
        "model_arch": "convnext",
        "model_path": "/home/sagemaker-user/lung-cancer/best_model.pth",
        "num_classes": 4,
        "classes": ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"],
        "class_info": {
            "adenocarcinoma":          ("Adenocarcinoma",           "Most common lung cancer type, often peripheral. Requires oncology referral."),
            "large_cell_carcinoma":    ("Large Cell Carcinoma",     "Aggressive undifferentiated cancer. Rapid oncology review strongly advised."),
            "normal":                  ("No Pathology Detected",    "Lung tissue appears within normal limits. Routine follow-up recommended."),
            "squamous_cell_carcinoma": ("Squamous Cell Carcinoma",  "Central lung cancer arising from bronchial epithelium. Oncology referral needed."),
        },
        "color": "#e8af34",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — cached so it only loads once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(disease_key):
    cfg = DISEASE_CONFIG[disease_key]
    arch = cfg["model_arch"]
    num_classes = cfg["num_classes"]
    model_path = cfg["model_path"]

    if not os.path.exists(model_path):
        return None, f"Model file not found: `{model_path}`"

    try:
        if arch == "convnext":
            model = timm.create_model(
                "convnext_base.fb_in22k_ft_in1k",
                pretrained=False,
                num_classes=num_classes,
            )
        elif arch == "vit":
            model = timm.create_model(
                "vit_base_patch16_224.augreg_in21k",
                pretrained=False,
                num_classes=num_classes,
            )
        else:
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

        state = torch.load(model_path, map_location=device)
        # Handle wrapped state dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, image: Image.Image, classes):
    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    pred_cls  = classes[pred_idx]
    conf      = float(probs[pred_idx])
    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}
    return pred_cls, conf, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BAR CHART (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def make_confidence_chart(all_probs, pred_cls, accent_color):
    labels = list(all_probs.keys())
    values = [v * 100 for v in all_probs.values()]
    colors = [accent_color if lbl == pred_cls else "#393836" for lbl in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#cdccca", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=8, b=8),
        height=max(160, len(labels) * 50),
        xaxis=dict(
            range=[0, 115],
            showgrid=False, showticklabels=False, zeroline=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(color="#cdccca", size=13),
        ),
        showlegend=False,
        font=dict(family="Satoshi, Inter, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="MedScan logo">
          <rect width="32" height="32" rx="8" fill="#01696f"/>
          <circle cx="16" cy="13" r="5" stroke="white" stroke-width="2" fill="none"/>
          <line x1="16" y1="18" x2="16" y2="26" stroke="white" stroke-width="2" stroke-linecap="round"/>
          <line x1="12" y1="22" x2="20" y2="22" stroke="white" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:#f0efed;">MedScan</div>
            <div style="font-size:0.72rem;color:#797876;letter-spacing:0.05em;text-transform:uppercase;">AI Diagnostic Tool</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Select Disease Domain")
    selected_disease = st.selectbox(
        "Disease domain",
        list(DISEASE_CONFIG.keys()),
        label_visibility="collapsed",
    )

    cfg = DISEASE_CONFIG[selected_disease]
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#797876;margin-top:0.5rem;line-height:1.6;">
        {cfg['description']}
    </div>
    <div style="margin-top:1rem;font-size:0.78rem;color:#5a5957;">
        <strong style="color:#797876;">Classes</strong><br/>
        {"  ·  ".join(cfg['classes'])}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#5a5957;line-height:1.7;">
        <strong style="color:#797876;">Device</strong><br/>
        {"CUDA (GPU)" if torch.cuda.is_available() else "CPU"}<br/><br/>
        <strong style="color:#797876;">Architecture</strong><br/>
        {cfg['model_arch'].upper()}<br/><br/>
        <strong style="color:#797876;">Classes</strong><br/>
        {cfg['num_classes']}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#5a5957;line-height:1.6;">
        ⚠️ For <strong>research & educational purposes only</strong>. Not a medical device. 
        Clinical decisions must be made by qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.5rem">
    <h1 style="font-size:1.6rem;font-weight:700;color:#f0efed;margin:0;">{selected_disease}</h1>
    <p style="color:#797876;font-size:0.9rem;margin-top:0.25rem;">{cfg['description']} — Upload an image to classify</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, load_error = load_model(selected_disease)

if load_error:
    st.error(f"**Model not loaded** — {load_error}\n\nMake sure the `.pth` file is at the correct path in `DISEASE_CONFIG`.", icon="⚠️")
    st.info("The UI is fully functional. Point `model_path` in `DISEASE_CONFIG` to your saved weights file.")

# Layout
col_upload, col_results = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Upload Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload scan",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True, caption=uploaded.name)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip"><div class="val">{image.size[0]}×{image.size[1]}</div><div class="key">Pixels</div></div>
            <div class="metric-chip"><div class="val">{uploaded.size // 1024} KB</div><div class="key">File size</div></div>
            <div class="metric-chip"><div class="val">{image.mode}</div><div class="key">Color mode</div></div>
        </div>
        """, unsafe_allow_html=True)

with col_results:
    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Prediction</p>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:3rem 2rem;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#393836" stroke-width="1.5" style="margin:0 auto 1rem">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p style="color:#5a5957;font-size:0.9rem;">Upload a scan image to get a prediction</p>
        </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:2rem;">
            <p style="color:#bb653b;">Model not loaded — configure the model path in DISEASE_CONFIG</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running inference…"):
            t0 = time.time()
            pred_cls, conf, all_probs = run_inference(model, image, cfg["classes"])
            elapsed_ms = (time.time() - t0) * 1000

        cls_name, cls_desc = cfg["class_info"].get(pred_cls, (pred_cls, ""))
        accent = cfg["color"]

        # Confidence colour
        if conf >= 0.85:
            conf_color = "#6daa45"
        elif conf >= 0.60:
            conf_color = "#e8af34"
        else:
            conf_color = "#dd6974"

        is_normal = "normal" in pred_cls.lower() or "notumor" in pred_cls.lower()

        st.markdown(f"""
        <div class="result-card">
            <span class="prediction-badge">{'✓ Normal' if is_normal else '⚠ Pathology detected'}</span>
            <div class="disease-name">{cls_name}</div>
            <div class="disease-desc">{cls_desc}</div>

            <div style="display:flex;align-items:flex-end;gap:1rem;margin-top:1.25rem;">
                <div>
                    <div class="confidence-big" style="color:{conf_color};">{conf*100:.1f}%</div>
                    <div class="confidence-label">Confidence score</div>
                </div>
                <div style="flex:1;text-align:right;">
                    <div style="font-size:0.8rem;color:#5a5957;">{elapsed_ms:.0f} ms inference</div>
                    <div style="font-size:0.8rem;color:#5a5957;">{pred_cls} · rank #1</div>
                </div>
            </div>

            <div class="warning-box">
                This result is generated by an AI model and must not be used for clinical diagnosis 
                without review by a qualified medical professional.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence breakdown chart
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">Confidence Breakdown</p>', unsafe_allow_html=True)
        fig = make_confidence_chart(all_probs, pred_cls, accent)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Ranked table
        with st.expander("All class probabilities", expanded=False):
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for rank, (cls, prob) in enumerate(sorted_probs, 1):
                name, _ = cfg["class_info"].get(cls, (cls, ""))
                bar_w = int(prob * 100)
                bar_color = accent if cls == pred_cls else "#393836"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;border-bottom:1px solid #2a2928;">
                    <span style="color:#5a5957;font-size:0.75rem;width:1rem;">#{rank}</span>
                    <span style="font-size:0.85rem;color:#cdccca;width:12rem;">{name}</span>
                    <div style="flex:1;background:#22211f;border-radius:4px;height:6px;">
                        <div style="width:{bar_w}%;background:{bar_color};height:6px;border-radius:4px;transition:width 0.5s ease;"></div>
                    </div>
                    <span style="font-size:0.85rem;font-weight:600;color:#f0efed;width:3.5rem;text-align:right;">{prob*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
