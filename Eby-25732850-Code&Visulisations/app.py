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
import re
import google.generativeai as genai
from fpdf import FPDF, XPos, YPos
import tempfile
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedScan — Multi-Agent Diagnostic AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS (Original Style Restored)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

html, body, [class*="css"] {
    font-family: 'Satoshi', 'Inter', sans-serif;
}

/* Hide default Streamlit branding but keep sidebar toggle */
#MainMenu, footer { display: none; }
header { background: transparent !important; padding: 0 !important; }

/* Make sidebar toggle visible */
button[kind="header"] { display: inline-block !important; }

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
    background: #22211f !important;
    border: 2px solid #4f98a3 !important;
    border-radius: 8px;
    color: #cdccca !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #6daa45 !important;
}
[data-testid="stSelectbox"] svg {
    fill: #cdccca !important;
}

/* Agent Pipeline Cards */
.agent-card {
    background: #1c1b19;
    border: 1px solid #2a2928;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.agent-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #22211f;
}
.agent-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.agent-name {
    font-weight: 700;
    color: #f0efed;
    font-size: 0.85rem;
}
.agent-status {
    font-size: 0.8rem;
    color: #797876;
}
.agent-detail {
    font-size: 0.75rem;
    color: #5a5957;
    margin-top: 0.1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE CONFIG
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
        "model_arch": "efficientnet_b0",
        "model_path": "/Users/ebbb/Desktop/UTS/S3/DL/Project/brain_best.pt",
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
    "🫁 Lung Pathology (X-Ray)": {
        "description": "Chest X-Ray multi-pathology classification (NIH ChestX-ray14)",
        "model_arch": "convnext",
        "model_path": "/Users/ebbb/Desktop/UTS/S3/DL/Project/chest_best.pt",
        "num_classes": 15,
        "classes": [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
        ],
        "class_info": {
            "Atelectasis": ("Atelectasis", "Partial or complete collapse of a lung or lobe."),
            "Cardiomegaly": ("Cardiomegaly", "Enlargement of the heart, often due to underlying conditions."),
            "Consolidation": ("Consolidation", "Lung tissue filled with liquid instead of air."),
            "Edema": ("Pulmonary Edema", "Excess fluid in the lungs."),
            "Effusion": ("Pleural Effusion", "Build-up of fluid between the layers of tissue that line the lungs."),
            "Emphysema": ("Emphysema", "Damaged air sacs in the lungs causing shortness of breath."),
            "Fibrosis": ("Pulmonary Fibrosis", "Lung tissue becomes damaged and scarred."),
            "Hernia": ("Hiatal Hernia", "Part of the stomach pushes up through the diaphragm."),
            "Infiltration": ("Infiltration", "Substance denser than air (pus, blood, protein) in the lung parenchyma."),
            "Mass": ("Mass", "Large lesion in the lung, potential malignancy."),
            "No Finding": ("No Finding", "No pathology detected in the scan."),
            "Nodule": ("Nodule", "Small rounded mass in the lung."),
            "Pleural_Thickening": ("Pleural Thickening", "Scarring of the lung lining."),
            "Pneumonia": ("Pneumonia", "Infection that inflames the air sacs in one or both lungs."),
            "Pneumothorax": ("Pneumothorax", "Collapsed lung due to air leaking into the space between lung and chest wall.")
        },
        "color": "#e8af34",
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────────
class VisionAgent:
    def __init__(self, disease_key):
        self.cfg = DISEASE_CONFIG[disease_key]
        self.model = self._load_model()
    
    def _load_model(self):
        arch, num_classes, model_path = self.cfg["model_arch"], self.cfg["num_classes"], self.cfg["model_path"]
        if not os.path.exists(model_path): return None
        try:
            if arch == "convnext": model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=num_classes)
            elif arch == "vit": model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False, num_classes=num_classes)
            elif arch == "efficientnet_b0": model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
            else: model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
            state = torch.load(model_path, map_location=device)
            if "model_state_dict" in state: state = state["model_state_dict"]
            elif "state_dict" in state: state = state["state_dict"]
            model.load_state_dict(state)
            model.to(device).eval()
            return model
        except: return None

    def run_inference(self, image: Image.Image):
        if self.model is None: return None
        img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {"pred_cls": self.cfg["classes"][pred_idx], "confidence": float(probs[pred_idx]), "all_probs": {cls: float(p) for cls, p in zip(self.cfg["classes"], probs)}, "pred_idx": pred_idx}

    def get_heatmap(self, image: Image.Image, pred_idx: int):
        if self.model is None: return None
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
        rgb_img = np.array(image.convert("RGB").resize((224, 224))) / 255.0
        arch = self.cfg["model_arch"]
        if arch == "convnext": target_layers = [self.model.stages[-1]]
        elif arch == "vit": target_layers = [self.model.blocks[-1].norm1]
        elif arch in ["efficientnet_b0", "mobilenetv3"]: target_layers = [self.model.conv_head]
        else: return None
        try:
            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                return result.transpose(2, 3).transpose(1, 2)
            with GradCAM(model=self.model, target_layers=target_layers, reshape_transform=reshape_transform if arch=="vit" else None) as cam:
                grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_idx)], aug_smooth=True, eigen_smooth=True)[0]
            heatmap = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True, colormap=17, image_weight=0.55)
            return Image.fromarray(heatmap)
        except: return None

class ReportAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate(self, domain, cls_name, cls_desc, conf):
        domain_clean = re.sub(r'[^\x00-\x7F]+', '', domain).strip()
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        
        prompt = f"""
        You are a highly skilled AI medical assistant. Generate a professional, structured medical report based on the following AI image classification results.
        
        Domain: {domain_clean}
        AI Finding: {cls_name}
        Description: {cls_desc}
        AI Confidence Level: {conf*100:.2f}%
        
        Strictly follow this structure for consistency:
        
        1. PATIENT INFORMATION
        - Patient Name: [Patient Name/ID Placeholder]
        - Date of Birth: [DOB Placeholder]
        - Report Date: {datetime.now().strftime("%B %d, %Y")}
        
        2. STUDY DETAILS
        - Modality: {domain_clean}
        - Clinical Indication: Automated screening for {domain_clean} pathologies.
        
        3. FINDINGS
        [A detailed clinical description of the observations based on the finding: {cls_name}. Mention {cls_desc}]
        
        4. IMPRESSION
        [A concise summary of the primary finding and its significance.]
        
        5. CONFIDENCE LEVEL
        - AI Confidence: {conf*100:.2f}%
        
        6. RECOMMENDATIONS
        [Clear next steps, emphasizing clinical correlation and physician review.]
        
        7. DISCLAIMER
        This report is generated by an Artificial Intelligence system for screening purposes. It does not constitute a definitive medical diagnosis. All medical decisions must be made by a qualified healthcare professional.
        
        Tone: Professional, objective, and cautious.
        """
        try:
            response = self.model.generate_content(prompt, safety_settings=safety_settings)
            return response.text
        except Exception as e:
            return f"Error generating report: {e}"

class DocumentAgent:
    @staticmethod
    def create_pdf(report_text, heatmap_img=None):
        pdf = FPDF()
        pdf.add_page()
        
        # --- Professional Header & Logo ---
        # Draw teal background rectangle
        pdf.set_fill_color(1, 105, 111) # #01696f
        pdf.rect(0, 0, 210, 40, 'F')
        
        # Draw Procedural Logo (similar to the SVG in the UI)
        # Rounded background for logo
        pdf.set_fill_color(255, 255, 255) # White background for logo
        pdf.set_draw_color(255, 255, 255)
        pdf.rect(15, 8, 24, 24, 'F')
        pdf.set_fill_color(1, 105, 111)
        pdf.rect(17, 10, 20, 20, 'F')
        
        # Cross/Stethoscope symbol inside logo
        pdf.set_draw_color(255, 255, 255)
        pdf.set_line_width(1.0)
        pdf.circle(27, 17, 4, 'D') # Circle
        pdf.line(27, 22, 27, 28) # Vertical line
        pdf.line(23, 25, 31, 25) # Horizontal line
        
        # Add "MedScan AI" Text
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 26)
        pdf.set_xy(45, 12)
        pdf.cell(0, 10, "MedScan AI", align='L')
        
        pdf.set_font("helvetica", "I", 10)
        pdf.set_xy(45, 22)
        pdf.cell(0, 10, "Automated Medical Imaging Diagnostic Report", align='L')
        
        pdf.set_xy(10, 50) # Move cursor below header
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", size=11)
        
        # --- Clean and Format Report Text ---
        clean_text = re.sub(r'#+\s*', '', report_text).replace('**', '').replace('*', '')
        clean_text = re.sub(r'---+', '', clean_text)
        
        safe_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, text=safe_text)
        
        if heatmap_img:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                heatmap_img.save(tmp, format="PNG")
                tmp_path = tmp.name
            
            pdf.ln(10)
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "Explainability (Grad-CAM Heatmap)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.image(tmp_path, w=100)
            os.remove(tmp_path)
            
        return pdf.output()

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def log_pipeline_step(placeholder, steps):
    html = '<div class="agent-card"><p style="font-size:0.72rem;font-weight:700;color:#797876;text-transform:uppercase;margin-bottom:0.4rem;">Agent Pipeline</p>'
    for s in steps:
        html += f'<div class="agent-step"><div class="agent-dot" style="background:{s["color"]};"></div><div style="flex:1;"><div class="agent-name">{s["name"]} <span class="agent-status">— {s["status"]}</span></div><div class="agent-detail">{s["detail"]}</div></div></div>'
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)

def make_confidence_chart(all_probs, pred_cls, accent_color):
    labels, values = list(all_probs.keys()), [v * 100 for v in all_probs.values()]
    colors = [accent_color if lbl == pred_cls else "#393836" for lbl in labels]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)), text=[f"{v:.1f}%" for v in values], textposition="outside", textfont=dict(color="#cdccca", size=12)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=60, t=8, b=8), height=max(160, len(labels)*50), xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#cdccca", size=13)), showlegend=False, font=dict(family="Satoshi, Inter, sans-serif"))
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
st.markdown(f'<div style="margin-bottom:1.5rem"><h1 style="font-size:1.6rem;font-weight:700;color:#f0efed;margin:0;">{selected_disease}</h1><p style="color:#797876;font-size:0.9rem;margin-top:0.25rem;">{cfg["description"]} — Upload an image to classify</p></div>', unsafe_allow_html=True)

vision_agent = VisionAgent(selected_disease)
# Use the hardcoded API key as in the previous version
report_agent = ReportAgent("AIzaSyCqPC8SI3Cg5ZzwIFGXy4WPZoQTTPqvHS8")

if "uploaded_image" not in st.session_state: st.session_state.uploaded_image = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None

def clear_upload():
    st.session_state.uploaded_image = None
    st.session_state.uploaded_filename = None
    if "report" in st.session_state: st.session_state.report = None
    if "heatmap" in st.session_state: st.session_state.heatmap = None

if st.session_state.uploaded_image is None:
    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;text-align:center;">Upload Image</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded = st.file_uploader("Upload scan", type=["jpg", "jpeg", "png", "bmp", "tiff"], label_visibility="collapsed")
        if uploaded:
            st.session_state.uploaded_image = Image.open(uploaded).convert("RGB")
            st.session_state.uploaded_filename = uploaded.name
            st.rerun()
else:
    if st.button("⬅ Back to Upload"):
        clear_upload()
        st.rerun()

    st.markdown("---")
    image, filename = st.session_state.uploaded_image, st.session_state.uploaded_filename
    col_results, col_images = st.columns([1.2, 1], gap="large")

    with col_results:
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Prediction Results</p>', unsafe_allow_html=True)
        
        if vision_agent.model is None:
            st.error("Model not loaded. Please check model path.")
        else:
            pipeline_log = st.empty()
            steps = [{"name": "Vision Agent", "status": "Inference...", "detail": f"Running {cfg['model_arch'].upper()}", "color": "#e8af34"}]
            log_pipeline_step(pipeline_log, steps)
            
            t0 = time.time()
            res = vision_agent.run_inference(image)
            elapsed = (time.time() - t0) * 1000
            
            steps[0] = {"name": "Vision Agent", "status": "Complete ✓", "detail": f"Predicted: {res['pred_cls']} ({elapsed:.0f}ms)", "color": "#6daa45"}
            steps.append({"name": "Explainability Agent", "status": "Generating Heatmap...", "detail": "Computing Grad-CAM", "color": "#e8af34"})
            log_pipeline_step(pipeline_log, steps)
            
            if "heatmap" not in st.session_state or st.session_state.get("last_filename") != filename:
                st.session_state.heatmap = vision_agent.get_heatmap(image, res['pred_idx'])
                st.session_state.last_filename = filename
            
            steps[1] = {"name": "Explainability Agent", "status": "Complete ✓", "detail": "Heatmap generated", "color": "#6daa45"}
            log_pipeline_step(pipeline_log, steps)

            cls_name, cls_desc = cfg["class_info"].get(res["pred_cls"], (res["pred_cls"], ""))
            accent = cfg["color"]
            
            if "normal" in res["pred_cls"].lower() or "notumor" in res["pred_cls"].lower(): st.success("✓ Normal")
            else: st.warning("⚠ Pathology detected")
            
            st.subheader(cls_name)
            st.write(cls_desc)
            
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{res['confidence']*100:.1f}%")
            c2.metric("Inference Time", f"{elapsed:.0f} ms")
            
            st.info("⚠️ This result is generated by an AI model and must not be used for clinical diagnosis.")
            
            st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">Confidence Breakdown</p>', unsafe_allow_html=True)
            st.plotly_chart(make_confidence_chart(res["all_probs"], res["pred_cls"], accent), use_container_width=True, config={"displayModeBar": False})

            with st.expander("All class probabilities"):
                sorted_p = sorted(res["all_probs"].items(), key=lambda x: x[1], reverse=True)
                for rank, (cls, prob) in enumerate(sorted_p, 1):
                    name, _ = cfg["class_info"].get(cls, (cls, ""))
                    st.markdown(f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;border-bottom:1px solid #2a2928;"><span style="color:#5a5957;font-size:0.75rem;width:1rem;">#{rank}</span><span style="font-size:0.85rem;color:#cdccca;width:12rem;">{name}</span><div style="flex:1;background:#22211f;border-radius:4px;height:6px;"><div style="width:{prob*100}%;background:{accent if cls==res["pred_cls"] else "#393836"};height:6px;border-radius:4px;"></div></div><span style="font-size:0.85rem;font-weight:600;color:#f0efed;width:3.5rem;text-align:right;">{prob*100:.2f}%</span></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin:1rem 0 0.5rem;">Generate Medical Report</p>', unsafe_allow_html=True)
            
            if st.button("Generate AI Medical Report"):
                steps.append({"name": "Report Agent", "status": "Thinking...", "detail": "Consulting Gemini 2.5", "color": "#e8af34"})
                log_pipeline_step(pipeline_log, steps)
                st.session_state.report = report_agent.generate(selected_disease, cls_name, cls_desc, res["confidence"])
                if not st.session_state.report.startswith("Error"):
                    steps[-1] = {"name": "Report Agent", "status": "Complete ✓", "detail": "Clinical narrative generated", "color": "#6daa45"}
                    log_pipeline_step(pipeline_log, steps)

            if st.session_state.get("report"):
                # If report already exists but agents aren't in steps (e.g. on rerun), add them
                if not any(s['name'] == "Report Agent" for s in steps):
                    steps.append({"name": "Report Agent", "status": "Complete ✓", "detail": "Clinical narrative restored from state", "color": "#6daa45"})
                    log_pipeline_step(pipeline_log, steps)

                if st.session_state.report.startswith("Error"): st.error(st.session_state.report)
                else:
                    st.success("Report generated successfully!")
                    with st.expander("📄 View Generated Report", expanded=True): st.markdown(st.session_state.report)
                    with st.spinner("Preparing PDF..."):
                        if not any(s['name'] == "Document Agent" for s in steps):
                            steps.append({"name": "Document Agent", "status": "Formatting...", "detail": "Sanitizing PDF document", "color": "#e8af34"})
                            log_pipeline_step(pipeline_log, steps)
                        
                        pdf_bytes = DocumentAgent.create_pdf(st.session_state.report, st.session_state.get("heatmap"))
                        
                        if steps[-1]['name'] == "Document Agent":
                            steps[-1] = {"name": "Document Agent", "status": "Complete ✓", "detail": "PDF ready for download", "color": "#6daa45"}
                            log_pipeline_step(pipeline_log, steps)
                    st.download_button("📥 Download Report as PDF", data=bytes(pdf_bytes), file_name="MedScan_Agent_Report.pdf", mime="application/pdf", type="primary")

    with col_images:
        st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#797876;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Scan & Explainability</p>', unsafe_allow_html=True)
        heatmap = st.session_state.get("heatmap")
        if heatmap:
            c_orig, c_heat = st.columns(2)
            c_orig.image(image.resize((224, 224)), caption="Original Scan", use_container_width=True)
            c_heat.image(heatmap, caption=f"Grad-CAM — {cls_name}", use_container_width=True)
        else: st.image(image, use_container_width=True, caption=filename)

        st.markdown(f'<div class="metric-row" style="margin-top: 0.5rem;"><div class="metric-chip"><div class="val">{image.size[0]}×{image.size[1]}</div><div class="key">Pixels</div></div><div class="metric-chip"><div class="val">{image.mode}</div><div class="key">Color mode</div></div></div>', unsafe_allow_html=True)
        if heatmap: st.markdown('<p style="font-size:0.82rem;color:#797876;margin-top:0.75rem;">Highlighted regions show model focus. Red = high attention.</p>', unsafe_allow_html=True)
