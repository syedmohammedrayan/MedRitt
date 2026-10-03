import os
import json
import logging
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase5")

# Mock the settings before importing main
import os
os.environ["GEMINI_API_KEY"] = "mock_key"
os.environ["LLM_PROVIDER"] = "template"
os.environ["DATA_DIR"] = "../test_data"
os.environ["UPLOADS_DIR"] = "../test_data/uploads"
os.environ["HEATMAPS_DIR"] = "../test_data/heatmaps"
os.environ["THUMBNAILS_DIR"] = "../test_data/thumbnails"

from main import app
from db.database import Base, get_db
from db.models import Result, Scan, User
import db.crud as crud

# Create test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///../test_phase5.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create the lifespan context manually if using TestClient
client = TestClient(app)

TEST_ASSETS_DIR = "../test_assets/jeevansh"

# 11. Test Startup & Health
def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        logger.info(f"Health Response: {data}")
        assert data["models"]["skin_cancer"] == "loaded"
        assert data["models"]["pneumonia"] == "loaded"
        assert data["models"]["brain_tumor"] == "loaded"
        assert data["models"]["bone_fracture"] == "loaded"

def login(client):
    db = TestingSessionLocal()
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    user = crud.get_user_by_username(db, "doctor")
    if not user:
        crud.create_user(db, username="doctor", hashed_password=pwd_context.hash("doctor123"), role="doctor", full_name="Dr. Test", specialization="", qualification="", department_id=None)
    db.close()

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "doctor", "password": "doctor123"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return token

def upload_scan(client, token, scan_type, file_path):
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
        data = {"scan_type": scan_type}
        response = client.post("/api/v1/scan/upload", headers=headers, data=data, files=files)
        assert response.status_code in (200, 201), f"Expected 200 or 201, got {response.status_code}: {response.text}"
        return response.json()["scan_id"]

def analyze_scan(client, token, scan_id):
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(f"/api/v1/scan/analyze/{scan_id}", headers=headers)
    return response

def test_inference_and_persistence():
    with TestClient(app) as client:
        token = login(client)
        
        tests = [
            ("skin_cancer", "skin_cancer_test.jpg", "classification"),
            ("pneumonia", "pneumonia_test.jpeg", "classification"),
            ("brain_tumor", "brain_tumor_test.jpg", "detection"),
            ("bone_fracture", "bone_fracture_test.jpg", "detection"),
        ]
        
        db = TestingSessionLocal()
        
        try:
            for scan_type, filename, expected_task_type in tests:
                logger.info(f"\n--- Testing {scan_type} ---")
                
                # Upload
                file_path = os.path.join(TEST_ASSETS_DIR, filename)
                scan_id = upload_scan(client, token, scan_type, file_path)
                logger.info(f"Uploaded {scan_type}, scan_id={scan_id}")
                
                # Analyze
                resp = analyze_scan(client, token, scan_id)
                assert resp.status_code == 200, f"Analyze failed: {resp.text}"
                data = resp.json()
                
                logger.info(f"API Response Task Type: {data.get('task_type')}")
                logger.info(f"API Response Model ID: {data.get('model_id')}")
                
                # Verify API Output
                if expected_task_type == "classification":
                    logger.info(f"API Top Label: {data['classification']['top_label']}")
                    logger.info(f"API Confidence: {data['classification']['confidence']}")
                    logger.info(f"API All Scores Keys: {list(data['classification']['all_scores'].keys())}")
                    logger.info(f"API Heatmap URL: {data['localization']['heatmap_url']}")
                else:
                    logger.info(f"API Detections Count: {len(data['localization']['bounding_boxes'])}")
                    if data['localization']['bounding_boxes']:
                        logger.info(f"First Detection: {data['localization']['bounding_boxes'][0]}")
                    logger.info(f"API Image Dimensions: {data['localization']['image_width']}x{data['localization']['image_height']}")
                    logger.info(f"API Overlay URL: {data['localization']['overlay_url']}")
                
                # Check DB directly
                db_result = db.query(Result).filter(Result.scan_id == scan_id).first()
                assert db_result is not None
                
                logger.info(f"DB Task Type: {db_result.task_type}")
                logger.info(f"DB Model ID: {db_result.model_id}")
                
                if expected_task_type == "classification":
                    assert db_result.task_type == "classification"
                    assert db_result.top_label == data['classification']['top_label']
                    assert abs(db_result.confidence - data['classification']['confidence']) < 1e-5
                    assert db_result.all_scores is not None
                    logger.info("Classification DB Verification: PASSED")
                else:
                    assert db_result.task_type == "detection"
                    assert db_result.severity is None
                    assert db_result.all_scores is None
                    logger.info(f"DB Severity: {db_result.severity}")
                    logger.info(f"DB All Scores: {db_result.all_scores}")
                    logger.info(f"DB Overlay Path: {db_result.overlay_path}")
                    assert db_result.image_width == data['localization']['image_width']
                    assert db_result.image_height == data['localization']['image_height']
                    logger.info("Detection DB Verification: PASSED")
                
                # Verify scan linkage
                db_scan = db.query(Scan).filter(Scan.id == scan_id).first()
                assert db_scan is not None
                assert db_scan.scan_type == scan_type
                assert db_scan.status == "analyzed"
                logger.info("Scan Relation Verification: PASSED")
            
            # Verify History
            logger.info("\n--- Verifying History Endpoint ---")
            headers = {"Authorization": f"Bearer {token}"}
            history_resp = client.get("/api/v1/history", headers=headers)
            assert history_resp.status_code == 200
            history_data = history_resp.json()
            assert len(history_data["scans"]) >= 4
            logger.info("History endpoint successfully returned all scans without crashing on nullable fields.")
            
        finally:
            db.close()

def test_failure_handling():
    with TestClient(app) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        
        # Analyze non-existent scan
        resp = client.post("/api/v1/scan/analyze/fake-id", headers=headers)
        assert resp.status_code == 404
        logger.info("Failure handling: Analyzing non-existent scan returns 404.")
        
        # Note: Testing bad image content is omitted here for brevity since it requires more complex mocking,
        # but 404 is a good start.
        
if __name__ == "__main__":
    test_health()
    test_inference_and_persistence()
    test_failure_handling()
    print("All Phase 5 verification tests passed.")
