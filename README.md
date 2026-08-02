# SkinLens-AI

A Streamlit web app for skin lesion risk estimation using a PyTorch binary classifier. The app supports user-uploaded checkpoints and a default trained checkpoint, validates incoming images, and aligns the inference pipeline with the Kaggle notebook training workflow.

## Project Summary

SkinLens-AI is designed to:

- Load a trained PyTorch checkpoint from a default location or from an uploaded `.pth` / `.pt` file
- Preprocess skin lesion images using the same transform parameters used in the Kaggle notebook pipeline
- Verify uploaded images to avoid meaningless predictions on documents, screenshots, or non-skin inputs
- Run inference using a baseline EfficientNet-B0 classifier and display probability and label results
- Save prediction outputs to `outputs/`

## Repository Structure

- `app.py` — Streamlit application UI and workflow control
- `predict.py` — Model checkpoint loading, state dict extraction, validation, and prediction
- `model.py` — `BaselineModel` architecture (EfficientNet-B0 fallback to simple CNN)
- `transforms.py` — Inference preprocessing with 224x224 resize and ImageNet normalization
- `models/` — Default checkpoint and model metadata
- `notebooks/` — Kaggle notebook assets used for training and pipeline alignment
- `uploads/` — Saved image uploads for inference
- `uploaded_models/` — Saved user-uploaded checkpoint files
- `outputs/` — Saved prediction summary results

## Key Features

- Default checkpoint fallback: uses `models/best_model.pth` when no upload is provided
- Checkpoint upload support: users can upload a custom `.pth` / `.pt` checkpoint
- Image validation: rejects documents/screenshots and images without sufficient skin content
- Notebook-aligned preprocessing: uses 224x224 resize, ImageNet mean/std normalization, and tensor conversion
- Model sanity check: validates loaded checkpoint with a dummy forward pass before inference

## How It Works

1. The app starts and initializes session state and working directories.
2. A user may upload a model checkpoint. If valid, the app loads it and performs a sanity check.
3. If no checkpoint is uploaded, the app attempts to load the default model from `models/best_model.pth`.
4. A user uploads an image, and the app validates the image content.
5. If the image passes validation and the model is loaded, the app runs inference and displays the result.
6. Prediction results are saved in `outputs/`.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd SkinLens-AI
```

2. Create a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install the needed packages manually:

```bash
pip install streamlit torch torchvision opencv-python pillow albumentations
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:

- Upload a trained model checkpoint (`.pth` or `.pt`) or let the app use the default checkpoint
- Upload a close-up image of a skin lesion
- Click **Run analysis** when the image is valid
- Review the prediction probability and label

## Model Details

- `BaselineModel` is an EfficientNet-B0 based binary classifier when `torchvision` is available
- Falls back to a lightweight CNN model when EfficientNet is unavailable
- The app expects a checkpoint with one of these formats:
  - `model_state_dict`
  - `state_dict`
  - `model` dictionary
  - bare state dict

## Inference Pipeline

- Load image with OpenCV and convert to RGB
- Apply `valid_transform` from `transforms.py`
- Run the model in `torch.inference_mode()`
- Convert logits to probability using `torch.sigmoid`
- Return label:
  - `Malignant` if probability >= 0.5
  - `Benign` otherwise

## Notebook Alignment

The `notebooks/` directory contains Kaggle notebook assets used for reference. Important alignment points include:

- `IMAGE_SIZE = 224`
- `IMAGENET_MEAN = (0.485, 0.456, 0.406)`
- `IMAGENET_STD = (0.229, 0.224, 0.225)`
- Validation transform: resize, normalize, tensor conversion

## Validation Rules

The app rejects an uploaded image when:

- It appears to be a document or screenshot
- It lacks sufficient skin-like regions
- The checkpoint fails to load or pass the sanity check

## Files to Review

- `app.py`
- `predict.py`
- `model.py`
- `transforms.py`
- `models/model_config.json`
- `notebooks/05-06-dataset-dataloader.ipynb`
- `project_documentation.txt`

## Notes

- This app is not a medical diagnosis tool. It provides a probabilistic model output only.
- Use a real close-up photo of a skin lesion for meaningful predictions.
- If the uploaded model is not compatible with the expected architecture, the app will report an error.

## License

Include your project license here.
"# SkinLens-AI" 
