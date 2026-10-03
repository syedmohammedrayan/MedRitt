import sys
import uuid
import json
from sqlalchemy.orm import Session
from backend.db.database import get_session_factory
from backend.db import crud
from backend.db.models import User, Result

def run_tests():
    SessionLocal = get_session_factory("sqlite:///backend/data/app.db")
    db = SessionLocal()
    
    try:
        # Check users intact
        users = db.query(User).all()
        print(f"Users found: {len(users)}")
        if len(users) == 0:
            print("ERROR: Users missing!")
            sys.exit(1)
            
        # Create a dummy user if needed or use existing
        user = users[0]
        
        # 1. Classification Test
        print("\n--- Testing Classification Persistence ---")
        scan_id_cls = str(uuid.uuid4())
        scan_cls = crud.create_scan(
            db=db, scan_id=scan_id_cls, user_id=user.id, filename="test_cls.png", 
            scan_type="skin_cancer", modality="Dermatoscopy", file_path="/fake"
        )
        
        all_scores = {"Melanoma": 0.8, "Benign": 0.2}
        result_cls = crud.create_result(
            db=db, scan_id=scan_id_cls, task_type="classification", model_id="skin_cancer_1.0",
            top_label="Melanoma", confidence=0.8, severity=None, all_scores=all_scores,
            localization_type="heatmap", overlay_path="/fake_heatmap.png"
        )
        
        # Read back
        res_back_cls = db.query(Result).filter(Result.scan_id == scan_id_cls).first()
        print(f"Task Type: {res_back_cls.task_type}")
        print(f"Top Label: {res_back_cls.top_label}")
        print(f"Confidence: {res_back_cls.confidence}")
        print(f"All Scores: {res_back_cls.all_scores}")
        print(f"Overlay Path: {res_back_cls.overlay_path}")
        
        # 2. Detection Test
        print("\n--- Testing Detection Persistence ---")
        scan_id_det = str(uuid.uuid4())
        scan_det = crud.create_scan(
            db=db, scan_id=scan_id_det, user_id=user.id, filename="test_det.png", 
            scan_type="brain_tumor", modality="MRI", file_path="/fake"
        )
        
        bboxes = [{"class": "brain_tumor", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]
        result_det = crud.create_result(
            db=db, scan_id=scan_id_det, task_type="detection", model_id="brain_tumor_1.0",
            top_label="brain_tumor", confidence=0.95, severity=None, all_scores=None,
            localization_type="bbox", bounding_boxes=bboxes, image_width=640, image_height=640,
            overlay_path="/fake_overlay.png"
        )
        
        # Read back
        res_back_det = db.query(Result).filter(Result.scan_id == scan_id_det).first()
        print(f"Task Type: {res_back_det.task_type}")
        print(f"Detections: {res_back_det.bounding_boxes}")
        print(f"Dimensions: {res_back_det.image_width}x{res_back_det.image_height}")
        print(f"Overlay Path: {res_back_det.overlay_path}")
        print(f"Severity: {res_back_det.severity} (Expected: None)")
        print(f"All Scores: {res_back_det.all_scores} (Expected: None)")

        # Cleanup
        db.delete(res_back_cls)
        db.delete(scan_cls)
        db.delete(res_back_det)
        db.delete(scan_det)
        db.commit()
        
    finally:
        db.close()

if __name__ == "__main__":
    run_tests()
