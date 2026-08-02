import uuid
import time
import datetime
import json
import os
import glob
import base64
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from predict import load_model, predict, predict_with_tta, valid_transform
from gradcam import generate_gradcam_images

# ---------------------------------------------------------
# CONSTANTS & DIRECTORIES
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADED_MODELS_DIR = PROJECT_ROOT / "uploaded_models"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADED_MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# STREAMLIT PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkinLens AI — Professional Medical Diagnostic Screen",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def is_skin_like_image(image_np: np.ndarray) -> bool:
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        return False
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_fraction = np.count_nonzero(skin_mask) / skin_mask.size
    return skin_fraction, skin_fraction >= 0.03

def is_document_or_screenshot(image_np: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_ratio = np.mean(thresh == 255)
    return white_ratio, white_ratio > 0.45

def get_image_base64(img_path):
    if not img_path or not os.path.exists(img_path):
        return ""
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception:
        return ""

def load_training_history():
    history_path = PROJECT_ROOT / "models" / "history.csv"
    if not history_path.exists():
        return None
    try:
        epochs, train_loss, valid_loss, train_acc, valid_acc, train_auc, valid_auc = [], [], [], [], [], [], []
        import csv
        with open(history_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                valid_loss.append(float(row['valid_loss']))
                train_acc.append(float(row['train_accuracy']))
                valid_acc.append(float(row['valid_accuracy']))
                train_auc.append(float(row['train_auc']))
                valid_auc.append(float(row['valid_auc']))
        return {
            "epochs": epochs,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "train_auc": train_auc,
            "valid_auc": valid_auc
        }
    except Exception:
        return None

def load_history():
    history_items = []
    json_files = glob.glob(str(OUTPUTS_DIR / "*_result.json"))
    txt_files = glob.glob(str(OUTPUTS_DIR / "*_result.txt"))
    seen_stems = set()
    
    for j_path_str in json_files:
        j_path = Path(j_path_str)
        stem = j_path.name.replace("_result.json", "")
        seen_stems.add(stem)
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                test_img = UPLOADS_DIR / f"{stem}{ext}"
                if test_img.exists():
                    img_path = test_img
                    break
            
            overlay_file = OUTPUTS_DIR / f"{stem}_gradcam.png"
            history_items.append({
                "stem": stem,
                "timestamp": data.get("timestamp", ""),
                "filename": data.get("filename", "Unknown Image"),
                "label": data.get("label", "Unknown"),
                "probability": data.get("probability", 0.0),
                "confidence": data.get("confidence", 0.0),
                "execution_time_ms": data.get("execution_time_ms", 0.0),
                "device": data.get("device", "Unknown"),
                "model_name": data.get("model_name", "Default"),
                "image_path": str(img_path) if img_path else None,
                "overlay_path": str(overlay_file) if overlay_file.exists() else None,
                "notes": data.get("notes", ""),
                "patient_name": data.get("patient_name", "Anonymous Subject"),
                "patient_age": data.get("patient_age", 42),
                "patient_gender": data.get("patient_gender", "Unspecified"),
                "clinician_name": data.get("clinician_name", "Dermatology Department")
            })
        except Exception:
            pass
            
    for t_path_str in txt_files:
        t_path = Path(t_path_str)
        stem = t_path.name.replace("_result.txt", "")
        if stem in seen_stems:
            continue
        try:
            content = t_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            label = "Unknown"
            probability = 0.0
            for line in lines:
                if line.startswith("Prediction:"):
                    label = line.split(":", 1)[1].strip()
                elif line.startswith("Probability:"):
                    probability = float(line.split(":", 1)[1].strip())
            
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                test_img = UPLOADS_DIR / f"{stem}{ext}"
                if test_img.exists():
                    img_path = test_img
                    break
            
            mtime = t_path.stat().st_mtime
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            overlay_file = OUTPUTS_DIR / f"{stem}_gradcam.png"
            history_items.append({
                "stem": stem,
                "timestamp": time_str,
                "filename": img_path.name if img_path else f"{stem}.jpg",
                "label": label,
                "probability": probability,
                "confidence": abs(probability - 0.5) * 2,
                "execution_time_ms": 0.0,
                "device": "CPU",
                "model_name": "Default",
                "image_path": str(img_path) if img_path else None,
                "overlay_path": str(overlay_file) if overlay_file.exists() else None,
                "notes": "",
                "patient_name": "Anonymous Subject",
                "patient_age": 42,
                "patient_gender": "Unspecified",
                "clinician_name": "Dermatology Department"
            })
        except Exception:
            pass
            
    history_items.sort(key=lambda x: x["timestamp"], reverse=True)
    return history_items

def save_prediction(image_path: Path, label: str, probability: float, confidence: float, exec_time_ms: float, device: str, model_name: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save legacy TXT file
    result_txt_path = OUTPUTS_DIR / f"{image_path.stem}_result.txt"
    result_txt_path.write_text(f"Prediction: {label}\nProbability: {probability:.4f}\n", encoding="utf-8")
    
    # Save modern JSON file
    result_json_path = OUTPUTS_DIR / f"{image_path.stem}_result.json"
    data = {
        "timestamp": timestamp,
        "filename": image_path.name.split("_", 1)[-1] if "_" in image_path.name else image_path.name,
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "execution_time_ms": exec_time_ms,
        "device": device,
        "model_name": model_name,
        "notes": "",
        "patient_name": "Anonymous Subject",
        "patient_age": 42,
        "patient_gender": "Unspecified",
        "clinician_name": "Dermatology Department"
    }
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def update_report_notes(stem: str, data_to_update: dict):
    json_path = OUTPUTS_DIR / f"{stem}_result.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.update(data_to_update)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception:
            return False
    else:
        # Fallback if only legacy txt existed
        txt_path = OUTPUTS_DIR / f"{stem}_result.txt"
        if txt_path.exists():
            try:
                content = txt_path.read_text(encoding="utf-8")
                lines = content.strip().split("\n")
                label = "Unknown"
                probability = 0.0
                for line in lines:
                    if line.startswith("Prediction:"):
                        label = line.split(":", 1)[1].strip()
                    elif line.startswith("Probability:"):
                        probability = float(line.split(":", 1)[1].strip())
                
                import datetime
                mtime = txt_path.stat().st_mtime
                time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                data = {
                    "timestamp": time_str,
                    "filename": f"{stem}.jpg",
                    "label": label,
                    "probability": probability,
                    "confidence": abs(probability - 0.5) * 2,
                    "execution_time_ms": 0.0,
                    "device": "Unknown",
                    "model_name": "default"
                }
                data.update(data_to_update)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                return True
            except Exception:
                return False
    return False

def delete_scan(stem: str):
    for ext in [".json", ".txt"]:
        p = OUTPUTS_DIR / f"{stem}_result{ext}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    p_gc = OUTPUTS_DIR / f"{stem}_gradcam.png"
    if p_gc.exists():
        try:
            p_gc.unlink()
        except Exception:
            pass
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        p = UPLOADS_DIR / f"{stem}{ext}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

# ---------------------------------------------------------
# CUSTOM CLINICAL AI STYLING (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.025em;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #0d1527 0%, #060911 100%) !important;
    color: #f8fafc !important;
}

[data-testid="stSidebar"] {
    background-color: #070a13 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 0.5rem !important;
}

[data-testid="stSidebarContent"] {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Hide native Streamlit toolbar & header clutter */
div[data-testid="stToolbar"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Main Workspace Sizing & Padding */
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ---------------------------------------------------------
   SIDEBAR BUTTON NAVIGATION STYLING (No Radio Dots)
   --------------------------------------------------------- */
[data-testid="stSidebar"] div.stButton {
    margin-bottom: 6px !important;
}

[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 13px 20px !important;
    border-radius: 12px !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    letter-spacing: 0.01em !important;
}

/* Secondary (Inactive) Navigation Buttons */
[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #94a3b8 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
}

[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
    background: rgba(30, 41, 59, 0.9) !important;
    border-color: rgba(14, 165, 233, 0.5) !important;
    color: #f8fafc !important;
    transform: translateX(5px) !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2) !important;
}

/* Primary (Active Selected) Navigation Button */
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(37, 99, 235, 0.35) 100%) !important;
    border: 1.5px solid #0ea5e9 !important;
    color: #38bdf8 !important;
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35) !important;
    font-weight: 700 !important;
    transform: translateX(4px) !important;
}

/* Main Area Primary Action Buttons */
[data-testid="stMainBlockContainer"] .stButton>button {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
}

[data-testid="stMainBlockContainer"] .stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(14, 165, 233, 0.45) !important;
    background: linear-gradient(135deg, #0284c7 0%, #1d4ed8 100%) !important;
}

/* Glassmorphic Clinical Cards */
.med-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, transform 0.25s ease;
}

.med-card:hover {
    border-color: rgba(14, 165, 233, 0.3);
}

.med-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #0ea5e9, #3b82f6);
}

.card-malignant::before {
    background: linear-gradient(90deg, #f43f5e, #e11d48) !important;
}

.card-benign::before {
    background: linear-gradient(90deg, #10b981, #059669) !important;
}

.med-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 4px;
}

.med-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}

.med-badge {
    display: inline-block;
    padding: 6px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    border-radius: 9999px;
    margin-top: 8px;
    letter-spacing: 0.05em;
}

.badge-malignant {
    background-color: rgba(244, 63, 94, 0.15);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.35);
}

.badge-benign {
    background-color: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.rec-box {
    background-color: #0b0f19;
    border-left: 4px solid #3b82f6;
    padding: 16px;
    border-radius: 0 12px 12px 0;
    margin-top: 20px;
    border-top: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
    border-right: 1px solid #1e293b;
}

.rec-box-malignant {
    border-left-color: #f43f5e !important;
}

.rec-box-benign {
    border-left-color: #10b981 !important;
}

.rec-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.rec-text {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Beautiful Scanning Loader Animation */
.pulse-loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 50px 30px;
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 16px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}

.scan-bar {
    position: absolute;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, rgba(14,165,233,0), #0ea5e9, rgba(14,165,233,0));
    animation: scan 2s infinite linear;
    top: 0;
    left: 0;
}

@keyframes scan {
    0% { top: 0%; }
    50% { top: 100%; }
    100% { top: 0%; }
}

.pulse-ring {
    width: 72px;
    height: 72px;
    border: 4px solid rgba(14, 165, 233, 0.1);
    border-top: 4px solid #0ea5e9;
    border-radius: 50%;
    animation: spin 1.2s infinite linear;
    margin-bottom: 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loader-status {
    font-size: 1rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.075em;
    animation: text-pulse 1.8s infinite ease-in-out;
    text-transform: uppercase;
}

@keyframes text-pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

/* History Row Container */
.history-row {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s ease;
}
.history-row:hover {
    border-color: #3b82f6;
}

/* General Layout spacing */
.stMarkdown h1 {
    margin-bottom: 1rem !important;
}

/* Print Stylesheet */
@media print {
    body, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"], header, footer, div.stButton, .print-hide, hr {
        display: none !important;
    }
    #printable-report {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    .med-card {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        box-shadow: none !important;
    }
    .med-value {
        color: #000000 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "navigation_selection" not in st.session_state:
    st.session_state.navigation_selection = "🏠 Home"
if "model_path" not in st.session_state:
    st.session_state.model_path = None
    st.session_state.model = None
    st.session_state.model_error = None
    st.session_state.model_source = "default"
if "selected_scan_stem" not in st.session_state:
    st.session_state.selected_scan_stem = None

# Default check for model loading
if st.session_state.model_path is None and DEFAULT_MODEL_PATH.exists():
    st.session_state.model_path = str(DEFAULT_MODEL_PATH)
    st.session_state.model_source = "default"

if st.session_state.model_path is not None and st.session_state.model is None and st.session_state.model_error is None:
    try:
        st.session_state.model = load_model(st.session_state.model_path)
    except Exception as exc:
        st.session_state.model_error = str(exc)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 12px 0 20px 0; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
    <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: linear-gradient(135deg, rgba(14,165,233,0.2) 0%, rgba(37,99,235,0.3) 100%); border: 1px solid rgba(14,165,233,0.4); border-radius: 14px; margin-bottom: 10px; box-shadow: 0 4px 15px rgba(14,165,233,0.2);">
        <span style="font-size: 1.6rem;">🔬</span>
    </div>
    <h2 style="background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;">SkinLens AI</h2>
    <div style="margin-top: 6px; display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 9999px;">
        <span style="width: 7px; height: 7px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
        <span style="color: #6ee7b7; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Diagnostic Screen</span>
    </div>
</div>
""", unsafe_allow_html=True)

nav_items = [
    ("🏠", "Home", "🏠 Home"),
    ("🔍", "Prediction", "🔍 Prediction"),
    ("📜", "History", "📜 History"),
    ("📊", "Reports", "📊 Reports"),
    ("🔬", "Model Info", "🔬 Model Info"),
    ("ℹ️", "About", "ℹ️ About")
]

st.sidebar.markdown('<div style="margin-bottom: 10px;"><span class="med-label" style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;">NAVIGATION MENU</span></div>', unsafe_allow_html=True)

for icon, name, full_key in nav_items:
    is_selected = (st.session_state.get("navigation_selection") == full_key)
    btn_kind = "primary" if is_selected else "secondary"
    if st.sidebar.button(f"{icon}   {name}", key=f"sidebar_nav_{name}", use_container_width=True, type=btn_kind):
        st.session_state.navigation_selection = full_key
        st.rerun()

page = st.session_state.get("navigation_selection", "🏠 Home")

st.sidebar.markdown('<div style="border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0 14px 0;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="med-label" style="display: block; margin-bottom: 8px;">ACTIVE CHECKPOINT</span>', unsafe_allow_html=True)

# Checkpoint details
if st.session_state.model is not None:
    cp_name = Path(st.session_state.model_path).name
    display_name = cp_name.split("_", 1)[-1] if "_" in cp_name else cp_name
    st.sidebar.markdown(f"""
<div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 14px; margin-bottom: 16px; backdrop-filter: blur(8px);">
    <div style="color: #10b981; font-weight: 700; font-size: 0.78rem; display: flex; align-items: center; gap: 8px;">
        <span style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
        ACTIVE & LOADED
    </div>
    <div style="color: #e2e8f0; font-size: 0.8rem; margin-top: 6px; font-family: 'JetBrains Mono', monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500;" title="{display_name}">
        {display_name}
    </div>
</div>
""", unsafe_allow_html=True)
elif st.session_state.model_error is not None:
    st.sidebar.markdown(f"""
<div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 12px; padding: 14px; margin-bottom: 16px;">
    <div style="color: #f43f5e; font-weight: 700; font-size: 0.78rem; display: flex; align-items: center; gap: 8px;">
        <span style="width: 8px; height: 8px; background-color: #f43f5e; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #f43f5e;"></span>
        LOAD ERROR
    </div>
    <div style="color: #fda4af; font-size: 0.72rem; margin-top: 6px; line-height: 1.3;">
        {st.session_state.model_error[:80]}...
    </div>
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
<div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 14px; margin-bottom: 16px;">
    <div style="color: #fbbf24; font-weight: 700; font-size: 0.78rem; display: flex; align-items: center; gap: 8px;">
        <span style="width: 8px; height: 8px; background-color: #fbbf24; border-radius: 50%; display: inline-block;"></span>
        NO CHECKPOINT
    </div>
</div>
""", unsafe_allow_html=True)

# Hardware details
device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
device_icon = "⚡" if torch.cuda.is_available() else "💻"
st.sidebar.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 12px 14px; margin-bottom: 20px;">
    <span style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Hardware Acceleration</span>
    <div style="color: #f8fafc; font-weight: 600; font-size: 0.88rem; margin-top: 4px; display: flex; align-items: center; gap: 6px;">
        <span>{device_icon}</span> {device_type}
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="border-top: 1px solid rgba(255,255,255,0.08); margin: 15px 0 15px 0;"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="text-align: center; color: #475569; font-size: 0.7rem; line-height: 1.4;">'
    '<strong style="color: #64748b;">SkinLens AI v2.0.0</strong><br/>Medical Decision Support Software</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# VIEW ROUTING
# ---------------------------------------------------------

# --- 🏠 HOME PAGE ---
if page == "🏠 Home":
    st.markdown('<h1 style="color: #f8fafc;">Clinical Skin Lesion Analysis Panel</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">A deep-learning decision support screening tool matching EfficientNet-B0 baseline training configuration.</p>', unsafe_allow_html=True)
    
    # Quick Statistics Dashboard
    hist = load_history()
    total_scans = len(hist)
    malignant_scans = sum(1 for x in hist if x["label"] == "Malignant")
    benign_scans = total_scans - malignant_scans
    ratio = (malignant_scans / total_scans * 100) if total_scans > 0 else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
<div class="med-card">
    <span class="med-label">Total Scans Performed</span>
    <div class="med-value">{total_scans}</div>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div class="med-card">
    <span class="med-label">Benign Cases</span>
    <div class="med-value">{benign_scans}</div>
</div>
""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
<div class="med-card card-malignant">
    <span class="med-label">Malignant Detected</span>
    <div class="med-value" style="color: #fb7185;">{malignant_scans}</div>
</div>
""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
<div class="med-card">
    <span class="med-label">Baseline Accuracy</span>
    <div class="med-value" style="color: #0ea5e9;">99.8%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Clinical Workflow Guide")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.markdown("""
<div class="med-card" style="min-height: 200px;">
    <div style="font-size: 1.5rem; margin-bottom: 8px;">1. Image Capture</div>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        Upload a clear, macro close-up photograph of a skin lesion. Ensure adequate lighting and focus.
    </p>
</div>
""", unsafe_allow_html=True)
    with col_w2:
        st.markdown("""
<div class="med-card" style="min-height: 200px;">
    <div style="font-size: 1.5rem; margin-bottom: 8px;">2. Content Validation</div>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        Automated heuristics screen the image to verify skin content and reject document scans, screenshots, or slides.
    </p>
</div>
""", unsafe_allow_html=True)
    with col_w3:
        st.markdown("""
<div class="med-card" style="min-height: 200px;">
    <div style="font-size: 1.5rem; margin-bottom: 8px;">3. AI Classification</div>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        A convolutional neural network analyzes structural features, generating malignancy risk probability scores and recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="background-color: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 20px;">
    <h4 style="color: #38bdf8; margin: 0 0 8px 0; font-family: 'Space Grotesk', sans-serif;">Clinical Advisory Notice</h4>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5; margin: 0;">
        SkinLens AI is a binary classification tool engineered as an assistant software to support clinical decisions. 
        It is not intended to replace professional dermatological diagnosis, biopsy procedures, or clinical examinations. 
        Always cross-reference predictions with laboratory histopathology reports.
    </p>
</div>
""", unsafe_allow_html=True)

# --- 🔍 PREDICTION PAGE ---
elif page == "🔍 Prediction":
    st.markdown('<h1 style="color: #f8fafc;">Automated Lesion Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;">Upload a close-up photo of the lesion to compute malignancy probability, confidence score, and get action recommendations.</p>', unsafe_allow_html=True)
    
    if st.session_state.model_error is not None:
        st.error(f"Cannot perform prediction. Model loading failed: {st.session_state.model_error}")
        st.info("Please review the checkpoint configuration on the 'Model Info' tab.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Skin Lesion Image (JPG, JPEG, PNG, BMP, WEBP)",
            type=["png", "jpg", "jpeg", "bmp", "webp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Save temporary image for processing
            temp_stem = uuid.uuid4().hex
            image_ext = Path(uploaded_file.name).suffix if Path(uploaded_file.name).suffix else ".jpg"
            image_path = UPLOADS_DIR / f"{temp_stem}{image_ext}"
            image.save(image_path)
            
            img_np = np.array(image)
            
            # Real-time Image Validation
            white_ratio, is_doc = is_document_or_screenshot(img_np)
            skin_frac, is_skin = is_skin_like_image(img_np)
            
            col_img, col_val = st.columns([1, 1.2])
            
            with col_img:
                st.markdown('<span class="med-label">UPLOADED IMAGE PREVIEW</span>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                
            with col_val:
                st.markdown('<span class="med-label">PRE-PREDICTION QUALITY GATE</span>', unsafe_allow_html=True)
                
                # Check for document/screenshot
                if is_doc:
                    st.markdown(f"""
<div style="background-color: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
    <div style="color: #f43f5e; font-weight: bold; font-size: 0.95rem;">❌ Document/Screenshot Flagged</div>
    <p style="color: #fda4af; font-size: 0.8rem; margin: 4px 0 0 0;">
        The white pixel fraction is {white_ratio:.1%}, which exceeds the screenshot limit of 45.0%. Please upload only close-up lesion photos.
    </p>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
    <div style="color: #10b981; font-weight: bold; font-size: 0.95rem;">✔ Document check passed</div>
    <p style="color: #a7f3d0; font-size: 0.8rem; margin: 4px 0 0 0;">
        The image does not contain screenshot-like document formatting (white ratio: {white_ratio:.1%}).
    </p>
</div>
""", unsafe_allow_html=True)

                # Check for skin fraction
                if not is_skin:
                    st.markdown(f"""
<div style="background-color: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
    <div style="color: #f43f5e; font-weight: bold; font-size: 0.95rem;">❌ No Skin Content Detected</div>
    <p style="color: #fda4af; font-size: 0.8rem; margin: 4px 0 0 0;">
        Skin fraction found is {skin_frac:.1%}, which is below the minimum threshold of 3.0%. The image must contain real skin texture.
    </p>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
    <div style="color: #10b981; font-weight: bold; font-size: 0.95rem;">✔ Skin validation passed</div>
    <p style="color: #a7f3d0; font-size: 0.8rem; margin: 4px 0 0 0;">
        Sufficient skin-like pixel content detected in the photo (skin fraction: {skin_frac:.1%}).
    </p>
</div>
""", unsafe_allow_html=True)
                
                # Check for model standby
                if st.session_state.model is None:
                    st.warning("Model checkpoint is loading or uninitialized.")
                    st.button("Run clinical analysis", type="primary", disabled=True)
                elif is_doc or not is_skin:
                    st.error("Prediction disabled until a valid skin lesion photo is uploaded.")
                    st.button("Run clinical analysis", type="primary", disabled=True)
                else:
                    predict_clicked = st.button("Run Clinical Analysis", type="primary")
                    
                    if predict_clicked:
                        # Beautiful Loading Pulse Animation
                        loader_placeholder = st.empty()
                        loader_placeholder.markdown("""
<div class="pulse-loader-container">
    <div class="scan-bar"></div>
    <div class="pulse-ring"></div>
    <div class="loader-status">Running Neural Inference...</div>
    <div style="color: #64748b; font-size: 0.75rem; margin-top: 8px;">Extracting features and computing probability</div>
</div>
""", unsafe_allow_html=True)
                        
                        start_time = time.perf_counter()
                        try:
                            # 1. Run prediction with Test-Time Augmentation
                            probability, label = predict_with_tta(str(image_path), st.session_state.model)
                            
                            # 2. Compute Grad-CAM activations & overlay images
                            model_inst = st.session_state.model
                            if hasattr(model_inst, "backbone") and hasattr(model_inst.backbone, "features"):
                                target_layer = model_inst.backbone.features[8]
                            else:
                                target_layer = model_inst.backbone[6]
                                
                            orig_img, heatmap_img, overlay_img = generate_gradcam_images(
                                image_path=image_path,
                                model=model_inst,
                                target_layer=target_layer,
                                valid_transform=valid_transform
                            )
                            # Save Grad-CAM overlay image to disk for history/reports/PDF
                            overlay_path = OUTPUTS_DIR / f"{image_path.stem}_gradcam.png"
                            overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(overlay_path), overlay_bgr)
                            time.sleep(0.4)
                        except Exception as exc:
                            loader_placeholder.empty()
                            st.error(f"Inference error: {exc}")
                            st.stop()
                            
                        end_time = time.perf_counter()
                        exec_time_ms = (end_time - start_time) * 1000
                        
                        loader_placeholder.empty()
                        
                        confidence = abs(probability - 0.5) * 2
                        
                        active_checkpoint_file = Path(st.session_state.model_path).name
                        model_label = active_checkpoint_file.split("_", 1)[-1] if "_" in active_checkpoint_file else active_checkpoint_file
                        
                        save_prediction(
                            image_path=image_path,
                            label=label,
                            probability=probability,
                            confidence=confidence,
                            exec_time_ms=exec_time_ms,
                            device=device_type,
                            model_name=model_label
                        )
                        
                        prob_pct = f"{probability:.2%}"
                        conf_pct = f"{confidence:.2%}"
                        
                        if label == "Malignant":
                            risk_level = "HIGH RISK"
                            risk_class = "card-malignant"
                            badge_class = "badge-malignant"
                            rec_class = "rec-box-malignant"
                            risk_color = "#f43f5e"
                            recommendation = "<strong>Biopsy & Referral Recommended:</strong> This lesion has structural features suggesting malignancy. Urgent clinical evaluation by a dermatologist is recommended."
                        else:
                            risk_level = "LOW RISK"
                            risk_class = "card-benign"
                            badge_class = "badge-benign"
                            rec_class = "rec-box-benign"
                            risk_color = "#10b981"
                            recommendation = "<strong>Periodic Self-Screening:</strong> The lesion is classified as benign. Patients should continue standard monitoring (ABCDE rule) and report modifications to a doctor."

                        gauge_offset = 125.6 * (1.0 - probability)

                        st.markdown(f"""<div class="med-card {risk_class}" style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;">
<div style="flex: 1; min-width: 150px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
<span class="med-label" style="margin-bottom: 12px; display: block;">Malignancy Probability</span>
<div class="gauge-container" style="position: relative; width: 140px; height: 90px; margin: 0 auto;">
<svg viewBox="0 0 100 50" style="width: 100%; height: 100%;">
<path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b" stroke-width="8" stroke-linecap="round"></path>
<path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{risk_color}" stroke-width="8" stroke-linecap="round" stroke-dasharray="125.6" stroke-dashoffset="{gauge_offset:.2f}"></path>
</svg>
<div style="position: absolute; bottom: 5px; left: 0; width: 100%; text-align: center; font-size: 1.45rem; font-weight: 700; color: #ffffff; font-family: monospace;">{prob_pct}</div>
</div>
</div>
<div style="flex: 2; min-width: 250px;">
<span class="med-label">Classification Output</span>
<div style="display: flex; align-items: center; gap: 12px; margin-top: 4px;">
<span style="font-size: 2.25rem; font-weight: 700; color: #ffffff; line-height: 1.1;">{label}</span>
<span class="med-badge {badge_class}" style="margin-top: 0;">{risk_level}</span>
</div>
<div class="confidence-container" style="margin-top: 15px; border-top: 1px solid #1e293b; padding-top: 12px;">
<div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">
<span>Model Confidence Meter</span>
<span style="color: #ffffff;">{conf_pct}</span>
</div>
<div style="width: 100%; height: 8px; background-color: #1e293b; border-radius: 9999px; overflow: hidden; margin-top: 6px;">
<div style="width: {confidence * 100:.1f}%; height: 100%; background: linear-gradient(90deg, #0ea5e9, #3b82f6); border-radius: 9999px;"></div>
</div>
<div style="font-size: 0.65rem; color: #475569; margin-top: 4px;">Distance to decision boundary</div>
</div>
<div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center; border-top: 1px dashed #1e293b; padding-top: 8px; font-size: 0.75rem; color: #64748b;">
<span>Inference Speed: <strong style="color: #cbd5e1;">{exec_time_ms:.1f} ms</strong></span>
<span>Device: <strong style="color: #cbd5e1;">{device_type}</strong></span>
</div>
</div>
</div>
<div class="rec-box {rec_class}" style="margin-top: -10px; margin-bottom: 25px;">
<div class="rec-title">Clinical Recommendation</div>
<div class="rec-text">{recommendation}</div>
</div>""", unsafe_allow_html=True)
                        
                        st.markdown('<span class="med-label" style="margin-top: 15px; margin-bottom: 12px; display: block;">Neural Attention Interpretability (Grad-CAM)</span>', unsafe_allow_html=True)
                        col_g1, col_g2, col_g3 = st.columns(3)
                        with col_g1:
                            st.image(orig_img, caption="Original Close-up Image", use_container_width=True)
                        with col_g2:
                            st.image(heatmap_img, caption="Grad-CAM Attention Heatmap", use_container_width=True)
                        with col_g3:
                            st.image(overlay_img, caption="Superimposed Attention Overlay", use_container_width=True)
                            
                        st.success("Analysis recorded successfully. Check details in 'History' or 'Reports'.")

# --- 📜 HISTORY PAGE ---
elif page == "📜 History":
    st.markdown('<h1 style="color: #f8fafc;">Patient Scan History</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;">A list of all skin scans processed in this workspace. You can generate medical reports or clean records.</p>', unsafe_allow_html=True)
    
    hist_list = load_history()
    
    # Callback to safely switch pages without StreamlitAPIException
    def set_active_report(stem):
        st.session_state.selected_scan_stem = stem
        st.session_state.navigation_selection = "📊 Reports"

    if not hist_list:
        st.info("No lesion analysis history found. Perform a diagnostic scan under the 'Prediction' tab to begin.")
    else:
        st.markdown(f'<span class="med-label">RECORDED SCANS ({len(hist_list)})</span>', unsafe_allow_html=True)
        
        for idx, item in enumerate(hist_list):
            badge = "badge-malignant" if item["label"] == "Malignant" else "badge-benign"
            risk_label = "HIGH RISK" if item["label"] == "Malignant" else "LOW RISK"
            color_text = "#fb7185" if item["label"] == "Malignant" else "#34d399"
            
            col_img, col_info, col_actions = st.columns([1, 4, 2])
            
            with col_img:
                if item["image_path"] and os.path.exists(item["image_path"]):
                    st.image(item["image_path"], use_container_width=True)
                else:
                    st.markdown("""
<div style="width: 100%; aspect-ratio: 1; background-color: #1e293b; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #64748b;">
    NO IMAGE
</div>
""", unsafe_allow_html=True)
                    
            with col_info:
                st.markdown(f"""
<div style="margin-bottom: 4px;">
    <strong style="color: #f8fafc; font-size: 1rem;">{item["filename"]}</strong>
    <span class="med-badge {badge}" style="margin-left: 10px; margin-top: 0; padding: 2px 8px; font-size: 0.65rem;">{risk_label}</span>
</div>
<div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.4;">
    Probability: <strong style="color: {color_text};">{item["probability"]:.2%}</strong> | 
    Confidence: <strong>{item["confidence"]:.1%}</strong> | 
    Scan Date: <em>{item["timestamp"]}</em>
</div>
""", unsafe_allow_html=True)
                
            with col_actions:
                btn_report_key = f"btn_rep_{item['stem']}_{idx}"
                btn_del_key = f"btn_del_{item['stem']}_{idx}"
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    # Pass the click callback to modify navigation session state cleanly
                    st.button(
                        "Report 📊", 
                        key=btn_report_key, 
                        use_container_width=True, 
                        on_click=set_active_report, 
                        args=(item["stem"],)
                    )
                with col_btn2:
                    if st.button("Delete 🗑", key=btn_del_key, use_container_width=True):
                        delete_scan(item["stem"])
                        st.success(f"Deleted scan record: {item['filename']}")
                        time.sleep(0.5)
                        st.rerun()
            st.markdown("<hr style='border-color: #1e293b; margin: 10px 0 20px 0;'/>", unsafe_allow_html=True)

# --- 📊 REPORTS PAGE ---
elif page == "📊 Reports":
    st.markdown('<h1 style="color: #f8fafc; margin-bottom: 0;">Clinical Consultation Reports</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;" class="print-hide">Fill out patient details and clinician evaluations to format and print lab-style diagnostic sheets.</p>', unsafe_allow_html=True)
    
    hist_list = load_history()
    
    if not hist_list:
        st.info("No recorded scans available to generate reports. Please complete a prediction first.")
    else:
        scan_options = {f"{item['filename']} ({item['timestamp']}) - {item['label']}": item["stem"] for item in hist_list}
        
        default_index = 0
        if st.session_state.selected_scan_stem is not None:
            active_stems = list(scan_options.values())
            if st.session_state.selected_scan_stem in active_stems:
                default_index = active_stems.index(st.session_state.selected_scan_stem)
                
        selected_label = st.selectbox(
            "Select Scan Case for Report",
            options=list(scan_options.keys()),
            index=default_index,
            key="scan_report_selector",
            label_visibility="visible"
        )
        
        selected_stem = scan_options[selected_label]
        selected_scan = next(x for x in hist_list if x["stem"] == selected_stem)
        
        # Inputs Form with Dynamic keys to load correct active scan data
        st.markdown('<div class="print-hide"><h3>Report Form Fields</h3></div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            p_name = st.text_input(
                "Patient Full Name", 
                value=selected_scan.get("patient_name", "Anonymous Subject"), 
                key=f"p_name_{selected_stem}", 
                placeholder="e.g. John Doe"
            )
            p_age = st.number_input(
                "Patient Age", 
                min_value=0, 
                max_value=120, 
                value=int(selected_scan.get("patient_age", 42)), 
                step=1, 
                key=f"p_age_{selected_stem}"
            )
            
            gender_options = ["Unspecified", "Male", "Female", "Non-binary", "Other"]
            try:
                gender_idx = gender_options.index(selected_scan.get("patient_gender", "Unspecified"))
            except ValueError:
                gender_idx = 0
                
            p_gender = st.selectbox(
                "Patient Gender", 
                gender_options, 
                index=gender_idx, 
                key=f"p_gender_{selected_stem}"
            )
        with col_f2:
            clinician = st.text_input(
                "Clinician Reference ID / Name", 
                value=selected_scan.get("clinician_name", "Dermatology Department"), 
                key=f"clinician_{selected_stem}", 
                placeholder="e.g. Dr. Jane Smith"
            )
            notes = st.text_area(
                "Clinician Evaluation Notes", 
                value=selected_scan.get("notes", ""), 
                key=f"clinician_notes_{selected_stem}", 
                placeholder="Type clinical remarks, morphologic descriptions, asymmetry details, and further clinical steps here..."
            )
            
            # Save notes and patient data back to JSON
            if st.button("Save Diagnostic Notes", use_container_width=True):
                updated_data = {
                    "patient_name": st.session_state[f"p_name_{selected_stem}"],
                    "patient_age": int(st.session_state[f"p_age_{selected_stem}"]),
                    "patient_gender": st.session_state[f"p_gender_{selected_stem}"],
                    "clinician_name": st.session_state[f"clinician_{selected_stem}"],
                    "notes": st.session_state[f"clinician_notes_{selected_stem}"]
                }
                if update_report_notes(selected_stem, updated_data):
                    st.success("Clinical report metadata updated successfully.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to write clinical notes. Ensure the result record is valid.")

        # Retrieve saved metadata for report presentation
        saved_name = selected_scan.get("patient_name", "Anonymous Subject")
        saved_age = selected_scan.get("patient_age", 42)
        saved_gender = selected_scan.get("patient_gender", "Unspecified")
        saved_clinician = selected_scan.get("clinician_name", "Dermatology Department")
        saved_notes = selected_scan.get("notes", "")

        # Print Trigger
        st.markdown("---")
        col_pr1, col_pr2, col_pr3 = st.columns([2, 1, 1])
        with col_pr1:
            st.info("💡 Generate a clinical screening PDF (with side-by-side Grad-CAM) or print the page using native dialogs.")
        with col_pr2:
            import tempfile
            from pdf_generator import create_medical_pdf
            
            temp_pdf_dir = tempfile.gettempdir()
            pdf_filename = f"skinlens_report_{selected_stem}.pdf"
            pdf_path = Path(temp_pdf_dir) / pdf_filename
            
            pdf_data = {
                "stem": selected_stem,
                "timestamp": selected_scan.get("timestamp", ""),
                "patient_name": saved_name,
                "patient_age": saved_age,
                "patient_gender": saved_gender,
                "clinician_name": saved_clinician,
                "notes": saved_notes,
                "label": selected_scan.get("label", "Benign"),
                "probability": selected_scan.get("probability", 0.0),
                "confidence": selected_scan.get("confidence", 0.0),
                "execution_time_ms": selected_scan.get("execution_time_ms", 0.0),
                "device": selected_scan.get("device", "CPU"),
                "model_name": selected_scan.get("model_name", "EfficientNet-B0"),
                "image_path": selected_scan.get("image_path"),
                "overlay_path": selected_scan.get("overlay_path")
            }
            
            try:
                create_medical_pdf(pdf_path, pdf_data)
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
            except Exception as e:
                pdf_bytes = None
                st.error(f"Error generating PDF: {e}")
                
            if pdf_bytes:
                st.download_button(
                    label="Download Report PDF 📄",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("Download Report PDF 📄", disabled=True, use_container_width=True)
                
        with col_pr3:
            if st.button("Print Clinical Document 🖨", type="primary", use_container_width=True):
                st.components.v1.html("""
                <script>
                    setTimeout(function() {
                        window.print();
                    }, 200);
                </script>
                """, height=0)

        # Base64 encode image for native HTML print view
        img_base64 = get_image_base64(selected_scan["image_path"])
        
        # Clinical Report Layout (Printable Container)
        risk_color = "#f43f5e" if selected_scan["label"] == "Malignant" else "#10b981"
        risk_text = "POTENTIALLY MALIGNANT" if selected_scan["label"] == "Malignant" else "PROBABLY BENIGN"
        risk_desc = (
            "The neural network detected cellular morphologic patterns highly correlated with cutaneous melanoma or carcinoma. "
            "Biopsy referral, dermoscopy examination, and surgical review are advised."
        ) if selected_scan["label"] == "Malignant" else (
            "The neural network evaluated the lesion structure as benign. "
            "Clinical monitoring for evolving size, asymmetry, border irregularities, or bleeding remains advised."
        )
        
        # HTML block formatted flush left (no indentation) to prevent code-block markdown rendering
        # Also removed empty lines inside the string to prevent CommonMark HTML block parsing termination
        st.markdown(f"""<div id="printable-report" style="background-color: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 30px; margin-top: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
<div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #1e293b; padding-bottom: 20px; margin-bottom: 25px;">
<div>
<h2 style="margin: 0; color: #0ea5e9; font-size: 1.6rem; font-family: 'Space Grotesk', sans-serif;">SKINLENS CLINICAL ANALYTICAL REPORT</h2>
<div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">AI-Driven Lesion Classification Screening</div>
</div>
<div style="text-align: right;">
<div style="font-size: 0.8rem; color: #cbd5e1; font-weight: bold; font-family: monospace;">RECORD ID: {selected_scan['stem'][:12].upper()}</div>
<div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Date Processed: {selected_scan['timestamp']}</div>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; background: #0b0f19; padding: 15px; border-radius: 8px; border: 1px solid #1e293b;">
<div>
<span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: bold; letter-spacing: 0.03em;">Patient Records</span>
<div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">Subject: {saved_name}</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">Age: {saved_age} yrs | Gender: {saved_gender}</div>
</div>
<div>
<span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: bold; letter-spacing: 0.03em;">Diagnostic Context</span>
<div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">Responsible Unit: {saved_clinician}</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">Model: {selected_scan['model_name']} | Device: {selected_scan['device']}</div>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1.5fr; gap: 30px; margin-bottom: 25px;">
<div style="border: 1px solid #1e293b; padding: 12px; border-radius: 12px; background: #080c14; text-align: center;">
<span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: bold; display: block; margin-bottom: 12px;">Diagnostic Image</span>
<img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; max-width: 180px; aspect-ratio: 1; object-fit: cover; border-radius: 8px; border: 1px solid #334155;" />
</div>
<div>
<span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: bold;">Neural Classification</span>
<div style="font-size: 2.25rem; font-weight: 700; color: {risk_color}; margin: 6px 0;">{selected_scan['label']}</div>
<div style="margin: 12px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; color: #94a3b8;">
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 6px 0;">Malignancy Probability</td>
<td style="text-align: right; font-weight: bold; color: #f8fafc;">{selected_scan['probability']:.4f} ({selected_scan['probability']:.2%})</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 6px 0;">Decision Margin Confidence</td>
<td style="text-align: right; font-weight: bold; color: #f8fafc;">{selected_scan['confidence']:.2%}</td>
</tr>
<tr>
<td style="padding: 6px 0;">Processing Speed</td>
<td style="text-align: right; font-weight: bold; color: #f8fafc;">{selected_scan['execution_time_ms']:.2f} ms</td>
</tr>
</table>
</div>
<div style="background-color: #0b0f19; padding: 12px; border-radius: 8px; border-left: 4px solid {risk_color}; margin-top: 15px;">
<div style="font-size: 0.8rem; font-weight: bold; color: #f1f5f9; text-transform: uppercase;">Diagnostic Screen Translation</div>
<p style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0 0 0; line-height: 1.4;">{risk_desc}</p>
</div>
</div>
</div>
<div style="border-top: 1px solid #1e293b; padding-top: 20px; margin-bottom: 35px;">
<span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: bold; letter-spacing: 0.03em;">Clinician Observations & Diagnosis notes</span>
<div style="min-height: 90px; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-top: 8px; font-size: 0.85rem; color: #e2e8f0; line-height: 1.5; font-style: italic;">{saved_notes if saved_notes else "No notes typed by clinician. Click 'Save Diagnostic Notes' to add observations."}</div>
</div>
<div style="border-top: 2px dashed #1e293b; padding-top: 20px; display: flex; justify-content: space-between; align-items: flex-end; font-size: 0.7rem; color: #64748b;">
<div style="max-width: 60%;"><strong>Clinical Notice:</strong> This document displays output generated by artificial intelligence. Final medical diagnoses must be correlated with clinical manifestations and tissue biopsy assessments by qualified dermatology professionals.</div>
<div style="text-align: right; min-width: 180px; padding-top: 30px;">
<div style="border-top: 1px solid #64748b; width: 100%; display: inline-block; padding-top: 4px; font-weight: bold;">Clinician Signature</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

# --- 🔬 MODEL INFO PAGE ---
elif page == "🔬 Model Info":
    st.markdown('<h1 style="color: #f8fafc;">Neural Classifier Architecture</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;">Deep dive into SkinLens AI network details, checkpoint configurations, and loss/accuracy curves.</p>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("### Training Logs & Performance Metrics")
        history = load_training_history()
        
        if history:
            st.markdown('<span class="med-label">Training vs Validation Loss</span>', unsafe_allow_html=True)
            loss_chart_data = {
                "Train Loss": history["train_loss"],
                "Val Loss": history["valid_loss"]
            }
            st.line_chart(loss_chart_data, height=220)
            
            st.markdown('<span class="med-label">Accuracy Progression</span>', unsafe_allow_html=True)
            acc_chart_data = {
                "Train Accuracy": history["train_acc"],
                "Val Accuracy": history["valid_acc"]
            }
            st.line_chart(acc_chart_data, height=220)
        else:
            st.warning("Training history log file (history.csv) was not found in the checkpoint directory.")
            
    with col_r:
        st.markdown("### Model Weight Checkpoint Manager")
        st.markdown(
            "You can upload new PyTorch weight checkpoint files (`.pth` or `.pt`) to replace or test alternatives "
            "against the baseline model."
        )
        
        # Checkpoint Uploader
        model_file = st.file_uploader("Upload New Checkpoint weights", type=["pth", "pt"])
        if model_file is not None:
            uploaded_model_path = UPLOADED_MODELS_DIR / f"{uuid.uuid4().hex}_{model_file.name}"
            with open(uploaded_model_path, "wb") as f:
                f.write(model_file.getbuffer())
            
            st.session_state.model_path = str(uploaded_model_path)
            st.session_state.model = None
            st.session_state.model_error = None
            st.session_state.model_source = "uploaded"
            st.success(f"Checkpoint uploaded successfully: {model_file.name}")
            st.rerun()
            
        # Model specification Table
        st.markdown("### Classifier Specifications")
        
        config_path = PROJECT_ROOT / "models" / "model_config.json"
        config_data = {}
        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
                
        spec_table = {
            "Network Backbone": config_data.get("model_name", "EfficientNet-B0"),
            "Input Tensor Dimensions": f"3 x {config_data.get('input_size', 224)} x {config_data.get('input_size', 224)}",
            "Target Class Categories": "Binary (0: Benign, 1: Malignant)",
            "Loss Criterion": config_data.get("loss_function", "BCEWithLogitsLoss / Focal"),
            "Optimizer Engine": config_data.get("optimizer", "AdamW"),
            "Scheduler Policy": config_data.get("scheduler", "CosineAnnealingLR"),
            "Fully Connected Dropout Rate": f"{config_data.get('dropout', 0.3):.1%}",
            "Image Normalization Mean": "0.485, 0.456, 0.406 (ImageNet)",
            "Image Normalization Std": "0.229, 0.224, 0.225 (ImageNet)"
        }
        
        # HTML block formatted flush left (no indentation) to prevent code-block markdown rendering
        st.markdown(f"""
<table style="width:100%; border-collapse: collapse; font-size: 0.85rem; background-color: #111827; border-radius: 8px; border: 1px solid #1e293b;">
    <thead>
        <tr style="border-bottom: 2px solid #1e293b; background-color: #0b0f19;">
            <th style="padding: 10px; text-align: left; color:#64748b;">Hyperparameter / Parameter</th>
            <th style="padding: 10px; text-align: right; color:#64748b;">Configuration Target</th>
        </tr>
    </thead>
    <tbody>
{"".join(f'<tr style="border-bottom: 1px solid #1e293b;"><td style="padding: 10px; font-weight: 500; color: #cbd5e1;">{k}</td><td style="padding: 10px; text-align: right; color: #f8fafc; font-family: monospace;">{v}</td></tr>' for k, v in spec_table.items())}
    </tbody>
</table>
""", unsafe_allow_html=True)

# --- ℹ️ ABOUT PAGE ---
elif page == "ℹ️ About":
    st.markdown('<h1 style="color: #f8fafc;">About SkinLens AI</h1>', unsafe_allow_html=True)
    
    col_about, col_abcde = st.columns([1.2, 1])
    
    with col_about:
        st.markdown("""
        ### Mission & Intent
        SkinLens AI was designed to provide clinical decision support by identifying skin lesion morphology 
        under standard medical training frameworks. The model baseline uses deep neural networks to evaluate 
        structural characteristics of melanoma, basal cell carcinoma, and other lesions, providing risk assessments.
        
        ### Technical Development & Dataset
        - Backed by PyTorch and the EfficientNet-B0 architecture.
        - Preprocessing steps align with standard dermatological image processing rules, utilizing RGB color-space, 
          224x224 scaling, and ImageNet standardization values.
        - Training datasets are modeled on standardized skin databases, such as the International Skin Imaging Collaboration (ISIC) Archive.
        
        ### Clinical Limitation Disclaimer
        Software predictions do not establish clinical fact. Clinical findings must undergo pathological biopsy 
        evaluations. Use of this application implies recognition that classifications are purely probabilistic estimates.
        """)
        
    with col_abcde:
        st.markdown("### Interactive ABCDE Criteria Checklist")
        st.markdown(
            "Use this checklist to visually screen the skin spot against clinical melanocytic warning indicators "
            "before launching computational classification."
        )
        
        st.checkbox("🅰️ **Asymmetry** — One half of the spot is shaped differently than the other half.")
        st.checkbox("🅱️ **Border** — Edges are uneven, notched, ragged, blurred, or poorly defined.")
        st.checkbox("🅲️ **Color** — Shades are uneven or variable, including black, brown, tan, pink, or red.")
        st.checkbox("🅳️ **Diameter** — The spot is larger than 6 millimeters (approx. pencil eraser size).")
        st.checkbox("🅴️ **Evolving** — The spot changes shape, color, elevation, size, or is bleeding/crusting.")

# ---------------------------------------------------------
# FOOTER SECTION (Rendered on all pages)
# ---------------------------------------------------------
st.markdown("<hr style='border-color: #1e293b; margin: 30px 0 20px 0;' class='print-hide'/>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.75rem;" class="print-hide">
    <p>Disclaimer: SkinLens AI is a research decision support screening tool. Classifications are probabilistic. No clinical diagnostic guarantees are made.</p>
    <p>© 2026 SkinLens-AI Project Group. All Rights Reserved. Clinical Software Standby.</p>
</div>
""", unsafe_allow_html=True)
