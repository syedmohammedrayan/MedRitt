"""
MedRittAI — FastAPI Application Entry Point
Main server with lifespan events, CORS, routing, and service initialization.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from db.database import init_db, get_session_factory
from db import crud

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Startup: create dirs, init DB, seed user, load ML models, init services.
    Shutdown: cleanup.
    """
    logger.info("=" * 60)
    logger.info("🏥 MedRittAI Diagnostic Engine — Starting Up")
    logger.info("=" * 60)

    # 1. Create data directories
    for dir_path in [settings.uploads_dir, settings.heatmaps_dir, settings.thumbnails_dir]:
        os.makedirs(dir_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    logger.info("📁 Data directories ready.")

    # 2. Initialize database
    init_db(settings.DATABASE_URL)
    logger.info("🗄️ Database initialized.")

    # 2b. Seed medicine catalog and pharmacy inventory if empty
    try:
        from medicine_catalog import MEDICINE_CATALOG
        from datetime import date, timedelta
        from passlib.context import CryptContext
        SessionLocal = get_session_factory()
        with SessionLocal() as db:
            medicines = crud.get_medicines(db)
            if not medicines:
                logger.info("💊 Seeding medicine catalog with standard medications...")
                crud.seed_medicine_catalog(db, MEDICINE_CATALOG)
                logger.info("✅ Medicine catalog seeded with %d items.", len(MEDICINE_CATALOG))
                medicines = crud.get_medicines(db)

            # Ensure default pharmacy user exists
            pharmacies = crud.get_users_by_role(db, "pharmacy")
            if not pharmacies:
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                pharm_user = crud.create_user(
                    db,
                    username="pharmacy",
                    hashed_password=pwd_context.hash("pharmacy123"),
                    role="pharmacy",
                    full_name="MedRitt Central Pharmacy",
                    email="pharmacy@medritt.ai",
                    phone="+91-800-PHARMACY",
                )
                pharmacies = [pharm_user]
                logger.info("✅ Default pharmacy account initialized.")

            # Seed initial stock for common medicines if inventory is empty
            pharmacy_id = pharmacies[0].id
            inv = crud.get_inventory(db, pharmacy_id)
            if not inv and medicines:
                logger.info("📦 Seeding initial stock for pharmacy store inventory...")
                for med in medicines[:30]:
                    crud.restock_inventory(
                        db,
                        pharmacy_id=pharmacy_id,
                        medicine_id=med.id,
                        quantity=100,
                        created_by_user_id=pharmacy_id,
                        expiry_date=date.today() + timedelta(days=365),
                    )
                logger.info("✅ Pharmacy inventory seeded with initial stock.")
    except Exception as seed_exc:
        logger.warning(f"Catalog seeding note: {seed_exc}")

    # 3. Load ML models

    # 4. Load ML models
    logger.info("🧠 Loading ML models...")

    # Initialize new Jeevansh Models (MUST fail if any checkpoint is missing)
    from services.jeevansh.model_registry import JeevanshModelRegistry
    app.state.jeevansh_registry = JeevanshModelRegistry(base_path=settings.resolve_path("models/jeevansh"))
    try:
        app.state.jeevansh_registry.load_all()
        logger.info("  ✅ All 4 Jeevansh models loaded successfully.")
    except Exception as exc:
        logger.error("  ❌ Failed to load Jeevansh models: %s", exc)
        raise RuntimeError(f"Jeevansh models failed to load: {exc}")

    # Legacy models have been removed from startup as per Phase 3 requirements.
    # The JeevanshModelRegistry now handles all active diagnostic inference.
    logger.info("  ✅ Model explainability engines initialized.")

    # 6. Initialize the independent, fail-closed scan type gate.
    from services.scan_type_verifier import ScanTypeVerifier
    app.state.scan_type_verifier = ScanTypeVerifier(
        api_key=settings.GEMINI_API_KEY,
        model=settings.SCAN_TYPE_VERIFIER_MODEL or settings.GEMINI_MODEL,
        min_confidence=settings.SCAN_TYPE_MIN_CONFIDENCE,
        groq_api_key=settings.GROQ_API_KEY,
        groq_model=settings.SCAN_TYPE_GROQ_MODEL,
        nvidia_api_key=settings.NVIDIA_API_KEY,
        nvidia_model=settings.NVIDIA_VISION_MODEL,
    )
    app.state.scan_verifier = app.state.scan_type_verifier
    logger.info("  ✅ Strict pre-inference scan type verification ready.")

    # 7. Initialize LLM Report Engine
    from services.llm_report_engine import LLMReportEngine
    app.state.report_engine = LLMReportEngine(
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_MODEL,
        sarvam_api_key=settings.SARVAM_API_KEY,
        sarvam_translate_model=settings.SARVAM_TRANSLATE_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        nvidia_api_key=settings.NVIDIA_API_KEY,
        nvidia_model=settings.NVIDIA_MODEL,
    )
    logger.info("  ✅ Clinical report and patient-language services ready.")

    # 8. Initialize PDF Generator
    from services.pdf_generator import PDFGenerator
    app.state.pdf_generator = PDFGenerator()
    logger.info("  ✅ PDF Generator ready.")

    # 9. Initialize AI Orchestrator (Phase 16)
    from services.ai_orchestrator import create_orchestrator
    app.state.orchestrator = create_orchestrator(app.state)
    logger.info("  ✅ AI Orchestrator initialized and ready.")

    logger.info("=" * 60)
    logger.info("MedRittAI backend is ready. Local frontend: http://localhost:5173")
    logger.info("=" * 60)

    yield  # App runs here

    # Shutdown
    app.state.report_engine.close()
    logger.info("🛑 MedRittAI shutting down.")



# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="MedRittAI API",
    description="AI-Powered Medical Image Diagnosis and Clinical Reporting Engine",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (uploads, heatmaps, thumbnails)
os.makedirs(settings.DATA_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.DATA_DIR), name="static")


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    registry = getattr(app.state, "jeevansh_registry", None)
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "models": {
            "skin_cancer": "loaded" if registry and "skin_cancer" in registry.models else "not_loaded",
            "pneumonia": "loaded" if registry and "pneumonia" in registry.models else "not_loaded",
            "brain_tumor": "loaded" if registry and "brain_tumor" in registry.models else "not_loaded",
            "bone_fracture": "loaded" if registry and "bone_fracture" in registry.models else "not_loaded",
        },
    }


# ============================================================
# REGISTER ROUTERS
# ============================================================

from routers import appointment, auth, case_study, diagnostic, doctor_reports, doctors, history, pharmacy, prescription, report, scan

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(scan.router, prefix="/api/v1/scan", tags=["Scan"])
app.include_router(report.router, prefix="/api/v1/report", tags=["Report"])
app.include_router(doctor_reports.router, prefix="/api/v1/reports", tags=["Doctor Reports"])
app.include_router(history.router, prefix="/api/v1/history", tags=["History"])
app.include_router(doctors.router, prefix="/api/v1", tags=["Hospital Directory"])
app.include_router(appointment.router, prefix="/api/v1/appointments", tags=["Appointments"])
app.include_router(diagnostic.router, prefix="/api/v1/diagnostic", tags=["Diagnostics"])
app.include_router(prescription.router, prefix="/api/v1/prescriptions", tags=["Prescriptions"])
app.include_router(pharmacy.router, prefix="/api/v1/pharmacy", tags=["Pharmacy"])
app.include_router(case_study.router, prefix="/api/v1/case-study", tags=["Case Studies"])
