from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

class BoneFractureDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        # Verify classes
        if self.model.names[0] != 'fracture':
            raise ValueError(f"Expected class 'fracture', got {self.model.names.get(0)}")
    
    def predict(self, image: Image.Image):
        # conf=0.25, iou=0.45, imgsz=1024
        results = self.model(image, conf=0.25, iou=0.45, imgsz=1024, verbose=False)
        result = results[0]
        
        detections = []
        best_conf = 0.0
        best_label = "No Detection"
        
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class": label, "confidence": conf
            })
            
            if conf > best_conf:
                best_conf = conf
                best_label = label
                
        overlay = self.generate_overlay(image, detections)
        
        return {
            "task_type": "detection",
            "model_id": "bone_fracture",
            "detections": detections,
            "image_width": image.width,
            "image_height": image.height,
            "overlay": overlay
        }

    def generate_overlay(self, image: Image.Image, detections: list) -> Image.Image:
        """
        Draws bounding boxes over the image.
        Note: This is an overlay, not a Grad-CAM heatmap.
        """
        img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
            conf = det["confidence"]
            label = det["class"]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
