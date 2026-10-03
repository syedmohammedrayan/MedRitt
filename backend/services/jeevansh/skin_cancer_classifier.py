import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import torchvision.transforms as transforms
from PIL import Image
from .gradcam import ClassifierGradCAM

class SkinCancerClassifier:
    CLASSES = [
        "Actinic_Keratoses",
        "Basal_Cell_Carcinoma",
        "Benign_Keratosis",
        "Dermatofibroma",
        "Melanocytic_Nevi",
        "Melanoma",
        "Vascular_Lesion"
    ]
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mobilenet_v3_large(weights=None)
        
        # Apply the exact custom classifier structure from the checkpoint
        self.model.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512, 7)
        )
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Verify class definitions match the checkpoint if available
        if "class_names" in ckpt and list(ckpt["class_names"]) != self.CLASSES:
            raise ValueError(f"Class mismatch. Expected {self.CLASSES}, got {ckpt['class_names']}")
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.gradcam = ClassifierGradCAM(self)

    def predict(self, image: Image.Image):
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
        conf, idx = torch.max(probs, 0)
        confidence = conf.item()
        top_label = self.CLASSES[idx.item()]
        
        all_scores = {self.CLASSES[i]: probs[i].item() for i in range(len(self.CLASSES))}
        gradcam = self.gradcam.generate_heatmap(image)
        
        return {
            "task_type": "classification",
            "model_id": "skin_cancer",
            "top_label": top_label,
            "confidence": confidence,
            "all_scores": all_scores,
            "gradcam": gradcam
        }
