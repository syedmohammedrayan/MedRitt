import os
import sys

# Ensure backend directory is in path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.jeevansh import JeevanshModelRegistry

def main():
    print("=== INITIALIZING JEEVANSH MODEL REGISTRY ===")
    registry = JeevanshModelRegistry(base_path="models/jeevansh")
    
    try:
        registry.load_all()
        print("\nAll models loaded successfully.\n")
    except Exception as e:
        print(f"\nERROR: Failed to load models.\nDetails: {e}")
        sys.exit(1)

    print("=== MODEL VERIFICATION ===")
    
    # 1. Skin Cancer
    print("\n[skin_cancer]")
    sc = registry.get_model("skin_cancer")
    print(f"  Architecture: {sc.model.__class__.__name__} (MobileNetV3)")
    print(f"  Task Type: Classification")
    print(f"  Input Size: 320x320 (Transform: {sc.transform.transforms[0].size})")
    print(f"  Classes: {sc.CLASSES}")
    
    # 2. Pneumonia
    print("\n[pneumonia]")
    pn = registry.get_model("pneumonia")
    print(f"  Architecture: {pn.model.__class__.__name__} (MobileNetV3)")
    print(f"  Task Type: Classification")
    print(f"  Input Size: 224x224 (Transform: {pn.transform.transforms[0].size})")
    print(f"  Classes: {pn.CLASSES}")
    
    # 3. Brain Tumor
    print("\n[brain_tumor]")
    bt = registry.get_model("brain_tumor")
    print(f"  Architecture: {bt.model.model.__class__.__name__} (YOLOv9m)")
    print(f"  Task Type: Object Detection")
    print(f"  Input Size: 640x640")
    print(f"  Classes: {bt.model.names}")
    
    # 4. Bone Fracture
    print("\n[bone_fracture]")
    bf = registry.get_model("bone_fracture")
    print(f"  Architecture: {bf.model.model.__class__.__name__} (YOLO11)")
    print(f"  Task Type: Object Detection")
    print(f"  Input Size: 1024x1024")
    print(f"  Classes: {bf.model.names}")

    print("\nSUCCESS: Verification complete.")

if __name__ == "__main__":
    main()
