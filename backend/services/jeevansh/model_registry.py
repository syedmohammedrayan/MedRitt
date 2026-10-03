import os
import logging
from typing import Dict, Any

from .skin_cancer_classifier import SkinCancerClassifier
from .pneumonia_classifier import PneumoniaClassifier
from .brain_tumor_detector import BrainTumorDetector
from .bone_fracture_detector import BoneFractureDetector
from .gradcam import ClassifierGradCAM

logger = logging.getLogger(__name__)

class JeevanshModelRegistry:
    def __init__(self, base_path: str = "models/jeevansh"):
        self.base_path = base_path
        self.models: Dict[str, Any] = {}
        self.gradcams: Dict[str, Any] = {}
        
    def load_all(self):
        logger.info(f"Loading Jeevansh models from {self.base_path}")
        
        # Define expected models
        configs = {
            "skin_cancer": {
                "class": SkinCancerClassifier,
                "filename": "skin_cancer.pth",
                "is_classifier": True
            },
            "pneumonia": {
                "class": PneumoniaClassifier,
                "filename": "pneumonia.pth",
                "is_classifier": True
            },
            "brain_tumor": {
                "class": BrainTumorDetector,
                "filename": "brain_tumour.pt",
                "is_classifier": False
            },
            "bone_fracture": {
                "class": BoneFractureDetector,
                "filename": "fracture.pt",
                "is_classifier": False
            }
        }
        
        for model_id, config in configs.items():
            model_path = os.path.join(self.base_path, config["filename"])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing checkpoint for {model_id}: {model_path}")
                
            try:
                logger.info(f"Loading {model_id}...")
                model_instance = config["class"](model_path)
                self.models[model_id] = model_instance
                
                if config["is_classifier"]:
                    self.gradcams[model_id] = ClassifierGradCAM(model_instance)
                    
            except Exception as e:
                logger.error(f"Failed to load {model_id} from {model_path}: {e}")
                raise RuntimeError(f"Failed to load {model_id} from {model_path}: {e}")
                
        logger.info("Successfully loaded all Jeevansh models.")
        
    def get_model(self, model_id: str):
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not loaded or not supported.")
        return self.models[model_id]
        
    def get_gradcam(self, model_id: str):
        if model_id not in self.gradcams:
            raise KeyError(f"GradCAM not available for {model_id}")
        return self.gradcams[model_id]
