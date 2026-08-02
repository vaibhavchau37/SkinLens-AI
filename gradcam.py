import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

class GradCAM:
    """
    Grad-CAM class to capture activations and gradients of target convolutional layers
    and generate attention heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, input_tensor):
        input_tensor.requires_grad_()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Score w.r.t output (binary logit)
        score = output[0, 0]
        
        # Backward pass
        score.backward()
        
        # Extract activations and gradients
        activations = self.activations.detach().cpu().numpy()[0]
        gradients = self.gradients.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combinations of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize heatmap to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Clean up hooks to prevent memory leaks
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        return cam

def generate_gradcam_images(image_path, model, target_layer, valid_transform):
    """
    Generates the original, heatmap attention map, and superimposed overlay images.
    """
    # Read original image in RGB
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Preprocess using same transform pipeline
    transformed = valid_transform(image=orig_img)
    image_tensor = transformed["image"].unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Run Grad-CAM in evaluation mode with gradients enabled
    model.eval()
    gradcam = GradCAM(model, target_layer)
    with torch.enable_grad():
        cam = gradcam(image_tensor)
        
    # Resize activation map to match original image dimensions
    h, w, _ = orig_img.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    # Scale heatmap to uint8
    heatmap = np.uint8(255 * cam_resized)
    
    # Apply colormap JET
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap with original image (alpha blend)
    alpha = 0.65
    overlay = cv2.addWeighted(orig_img, alpha, heatmap_colored, 1.0 - alpha, 0)
    
    return orig_img, heatmap_colored, overlay
