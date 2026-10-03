import os
import sys
from PIL import Image

# Ensure backend directory is in path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.jeevansh import JeevanshModelRegistry

def main():
    print("=== STARTING JEEVANSH INFERENCE SMOKE TEST ===\n")
    
    registry = JeevanshModelRegistry(base_path="models/jeevansh")
    registry.load_all()
    
    test_dir = "test_assets/jeevansh"
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = {
        "skin_cancer": os.path.join(test_dir, "skin_cancer_test.jpg"),
        "pneumonia": os.path.join(test_dir, "pneumonia_test.jpeg"),
        "brain_tumor": os.path.join(test_dir, "brain_tumor_test.jpg"),
        "bone_fracture": os.path.join(test_dir, "bone_fracture_test.jpg")
    }

    print("\n--- MODEL 1: skin_cancer ---")
    try:
        sc_model = registry.get_model("skin_cancer")
        img = Image.open(test_images["skin_cancer"])
        top_label, conf, all_scores = sc_model.predict(img)
        print(f"Predicted class: {top_label}")
        print(f"Confidence: {conf:.4f}")
        print("All probabilities:")
        for cls_name, prob in all_scores.items():
            print(f"  {cls_name}: {prob:.4f}")
            
        gradcam_engine = registry.get_gradcam("skin_cancer")
        heatmap = gradcam_engine.generate_heatmap(img)
        output_path = os.path.join(output_dir, "skin_cancer_gradcam.jpg")
        heatmap.save(output_path)
        print(f"Grad-CAM saved to: {output_path}")
        print(f"Grad-CAM size: {heatmap.size}")
    except Exception as e:
        print(f"FAIL skin_cancer: {e}")

    print("\n--- MODEL 2: pneumonia ---")
    try:
        pn_model = registry.get_model("pneumonia")
        img = Image.open(test_images["pneumonia"])
        top_label, conf, all_scores = pn_model.predict(img)
        print(f"Predicted class: {top_label}")
        print(f"Confidence: {conf:.4f}")
        print("All probabilities:")
        for cls_name, prob in all_scores.items():
            print(f"  {cls_name}: {prob:.4f}")
            
        gradcam_engine = registry.get_gradcam("pneumonia")
        heatmap = gradcam_engine.generate_heatmap(img)
        output_path = os.path.join(output_dir, "pneumonia_gradcam.jpg")
        heatmap.save(output_path)
        print(f"Grad-CAM saved to: {output_path}")
        print(f"Grad-CAM size: {heatmap.size}")
    except Exception as e:
        print(f"FAIL pneumonia: {e}")

    print("\n--- MODEL 3: brain_tumor ---")
    try:
        bt_model = registry.get_model("brain_tumor")
        img = Image.open(test_images["brain_tumor"])
        best_label, best_conf, detections = bt_model.predict(img)
        print(f"Number of detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"Detection {i+1}: Class={det['label']} Conf={det['confidence']:.4f} Box=[{det['x1']:.1f}, {det['y1']:.1f}, {det['x2']:.1f}, {det['y2']:.1f}]")
            
        overlay = bt_model.generate_overlay(img, detections)
        output_path = os.path.join(output_dir, "brain_tumor_overlay.jpg")
        overlay.save(output_path)
        print(f"Overlay saved to: {output_path}")
        print(f"Overlay size: {overlay.size}")
    except Exception as e:
        print(f"FAIL brain_tumor: {e}")

    print("\n--- MODEL 4: bone_fracture ---")
    try:
        bf_model = registry.get_model("bone_fracture")
        img = Image.open(test_images["bone_fracture"])
        best_label, best_conf, detections = bf_model.predict(img)
        print(f"Number of detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"Detection {i+1}: Class={det['label']} Conf={det['confidence']:.4f} Box=[{det['x1']:.1f}, {det['y1']:.1f}, {det['x2']:.1f}, {det['y2']:.1f}]")
            
        overlay = bf_model.generate_overlay(img, detections)
        output_path = os.path.join(output_dir, "bone_fracture_overlay.jpg")
        overlay.save(output_path)
        print(f"Overlay saved to: {output_path}")
        print(f"Overlay size: {overlay.size}")
    except Exception as e:
        print(f"FAIL bone_fracture: {e}")

if __name__ == '__main__':
    main()
