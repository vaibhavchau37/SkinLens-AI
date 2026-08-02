from pathlib import Path
import importlib.util
import sys

import cv2
import torch

try:
    from transforms import valid_transform
except Exception as exc:
    transforms_path = Path(__file__).resolve().with_name("transforms.py")
    spec = importlib.util.spec_from_file_location("transforms_module", transforms_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "valid_transform"):
        raise ImportError(f"Unable to load transforms from {transforms_path}: {exc}") from exc

    valid_transform = module.valid_transform

try:
    from model import BaselineModel
except Exception:
    model_path = Path(__file__).resolve().with_name("model.py")
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "BaselineModel"):
        raise ImportError(f"Unable to load BaselineModel from {model_path}")

    BaselineModel = module.BaselineModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        return checkpoint
    return checkpoint


def load_model(checkpoint_path, device=None):
    if device is None:
        device = DEVICE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid state dictionary")

    model = BaselineModel(pretrained=False)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: model state dict did not fully match the expected architecture. "
            f"missing_keys={load_result.missing_keys}, unexpected_keys={load_result.unexpected_keys}. "
            "Make sure the checkpoint was saved from the same model architecture."
        )
    model.to(device)
    model.eval()
    return model


def predict(image_path, model):
    if model is None:
        raise ValueError("A loaded model instance is required for prediction.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = valid_transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(next(model.parameters()).device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits).squeeze().item()

    label = "Malignant" if probability >= 0.5 else "Benign"
    return probability, label
