import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from predict import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADED_MODELS_DIR = PROJECT_ROOT / "uploaded_models"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"


def is_skin_like_image(image_np: np.ndarray) -> bool:
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        return False

    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_fraction = np.count_nonzero(skin_mask) / skin_mask.size
    return skin_fraction >= 0.03


def is_document_or_screenshot(image_np: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_ratio = np.mean(thresh == 255)
    return white_ratio > 0.45
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADED_MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="SkinLens-AI", page_icon="🧴", layout="centered")

if "model_path" not in st.session_state:
    st.session_state.model_path = None
    st.session_state.model = None
    st.session_state.model_error = None
    st.session_state.model_source = "default"

st.title("SkinLens-AI")
st.caption("Upload a skin lesion image to receive a quick malignancy risk estimate.")
st.markdown(
    "**Important:** only upload a close-up photo of a skin lesion.\n"
    "Do not upload screenshots, documents, posters, logos, or unrelated graphics."
)

model_file = st.file_uploader("Upload your trained model checkpoint", type=["pth", "pt"])

if model_file is not None:
    uploaded_model_path = UPLOADED_MODELS_DIR / f"{uuid.uuid4().hex}_{model_file.name}"
    with open(uploaded_model_path, "wb") as f:
        f.write(model_file.getbuffer())
    st.session_state.model_path = str(uploaded_model_path)
    st.session_state.model = None
    st.session_state.model_error = None
    st.session_state.model_source = "uploaded"
    st.success(f"Model uploaded: {uploaded_model_path.name}")

if st.session_state.model_path is None and DEFAULT_MODEL_PATH.exists():
    st.session_state.model_path = str(DEFAULT_MODEL_PATH)
    st.session_state.model_source = "default"

if st.session_state.model_path is not None and st.session_state.model is None and st.session_state.model_error is None:
    try:
        st.session_state.model = load_model(st.session_state.model_path)
        st.success("Model checkpoint loaded and sanity-checked successfully.")
    except Exception as exc:
        st.session_state.model_error = str(exc)
        st.error(f"Failed to load model checkpoint: {exc}")

if st.session_state.model_path is not None:
    source = "Uploaded model" if st.session_state.model_source == "uploaded" else "Default trained model"
    st.info(f"{source} ready: {Path(st.session_state.model_path).name}")
elif st.session_state.model_error is None:
    st.warning("Please upload a model checkpoint before running image predictions.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_path = UPLOADS_DIR / f"{uuid.uuid4().hex}_{uploaded_file.name}"
    image.save(image_path)

    st.image(image, caption="Uploaded image", use_container_width=True)

    img_np = np.array(image)
    is_document = is_document_or_screenshot(img_np)
    is_skin = is_skin_like_image(img_np)

    if is_document:
        st.error(
            "This upload looks like a screenshot, document, or non-skin image. "
            "Please upload only a close-up photo of a real skin lesion."
        )
    elif not is_skin:
        st.error(
            "This image does not appear to contain skin. "
            "Please upload only a close-up photo of a skin lesion."
        )
    else:
        st.success("Uploaded image passes skin-content validation.")

    if st.session_state.model_error is not None:
        st.error("Model checkpoint is invalid. Upload a compatible checkpoint before predicting.")
    elif st.session_state.model is None:
        st.info("Waiting for the model checkpoint to load.")
    elif is_document or not is_skin:
        st.warning("Prediction is disabled until a valid skin lesion image is uploaded.")
        st.button("Run analysis", type="primary", disabled=True)
    else:
        if st.button("Run analysis", type="primary"):
            with st.spinner("Analyzing the image..."):
                try:
                    probability, label = predict(str(image_path), st.session_state.model)
                except Exception as exc:
                    st.error(f"Unable to analyze the image: {exc}")
                    st.stop()

            st.success("Analysis complete")
            st.metric("Predicted probability", f"{probability:.2%}")
            st.metric("Prediction", label)

            if label == "Malignant":
                st.warning(
                    "This image is flagged as potentially malignant. Please consult a medical professional for a formal diagnosis."
                )
            else:
                st.info("This image appears more likely benign, but this is not a medical diagnosis.")

            result_path = OUTPUTS_DIR / f"{image_path.stem}_result.txt"
            result_path.write_text(f"Prediction: {label}\nProbability: {probability:.4f}\n", encoding="utf-8")
            st.caption(f"Saved result to {result_path}")
