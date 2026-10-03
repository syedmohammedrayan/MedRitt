import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

class ClassifierGradCAM:
    """Generates Grad-CAM overlays for MobileNetV3-Large models."""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.device = classifier.device
        self.model = classifier.model
        # For MobileNetV3, we typically target the last convolutional layer in features
        self.target_layers = [self.model.features[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, image: Image.Image, target_category: int = None) -> Image.Image:
        """
        Generate a Grad-CAM heatmap overlay.
        If target_category is None, the highest scoring category is used.
        """
        img_rgb = image.convert("RGB")
        input_tensor = self.classifier.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Calculate image input size to ensure correct overlay scale
        img_size = input_tensor.shape[-2:] # (H, W)
        
        # Resize original image to the model's input size for the overlay
        img_resized = img_rgb.resize((img_size[1], img_size[0]), Image.Resampling.LANCZOS)
        img_normalized = np.float32(img_resized) / 255.0
        
        # Generate the CAM mask
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create the visual overlay
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        return Image.fromarray(cam_image)
