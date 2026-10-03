import sys
from PIL import Image
from services.jeevansh.model_registry import JeevanshModelRegistry

def test():
    # Load registry
    registry = JeevanshModelRegistry(base_path="../models/jeevansh")
    registry.load_all()
    
    # Create a dummy image
    img = Image.new('RGB', (640, 640), color = 'white')
    
    # Test all 4 models
    for scan_type in ["skin_cancer", "pneumonia", "brain_tumor", "bone_fracture"]:
        print(f"\n--- Testing {scan_type} ---")
        model = registry.get_model(scan_type)
        res = model.predict(img)
        print(f"Task Type: {res.get('task_type')}")
        print(f"Model ID: {res.get('model_id')}")
        if res.get("task_type") == "classification":
            print(f"Top Label: {res.get('top_label')}")
            print(f"Confidence: {res.get('confidence')}")
            print(f"All Scores Keys: {list(res.get('all_scores', {}).keys())}")
            print(f"Has Grad-CAM: {'gradcam' in res and res['gradcam'] is not None}")
        elif res.get("task_type") == "detection":
            print(f"Detections Count: {len(res.get('detections', []))}")
            if res.get('detections'):
                print(f"First Detection Keys: {list(res.get('detections')[0].keys())}")
            print(f"Image WxH: {res.get('image_width')}x{res.get('image_height')}")
            print(f"Has Overlay: {'overlay' in res and res['overlay'] is not None}")

if __name__ == '__main__':
    test()
