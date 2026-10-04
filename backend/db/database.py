"""
MedRittAI - Database Engine & Session Management
SQLite via SQLAlchemy 2.0 ORM
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os
import sys

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass

_engine = None
_SessionLocal = None

import logging
logger = logging.getLogger(__name__)

def get_engine(database_url: str = None):
    global _engine
    if _engine is not None:
        return _engine

    from config import settings

    primary_url = database_url or settings.DATABASE_URL

    if not primary_url:
        logger.error("DATABASE_URL is completely blank or missing! Startup failed.")
        sys.exit(1)

    logger.info(f"database_backend=sqlite")

    connect_args = {}
    
    if primary_url.startswith("postgres://"):
        primary_url = primary_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif primary_url.startswith("postgresql://"):
        primary_url = primary_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if "sqlite:///" in primary_url:
        path = primary_url.split("sqlite:///")[-1]
        if path.startswith("./"):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        connect_args = {"check_same_thread": False}

    _engine = create_engine(primary_url, echo=False, pool_pre_ping=True, connect_args=connect_args)

    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"SQLite connection failed: {e}")
        sys.exit(1)

    return _engine

def get_session_factory(database_url: str = None):
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal

def get_db():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(database_url: str = None):
    from . import models as _
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    _apply_compatibility_migrations(engine)

def _apply_compatibility_migrations(engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    additions = {
        "users": {
            "role": "VARCHAR(20) NOT NULL DEFAULT 'patient'",
            "full_name": "VARCHAR(150) DEFAULT ''",
            "email": "VARCHAR(150) DEFAULT ''",
            "phone": "VARCHAR(20) DEFAULT ''",
            "specialization": "VARCHAR(100) DEFAULT ''",
            "qualification": "VARCHAR(150) DEFAULT ''",
            "department_id": "INTEGER REFERENCES departments(id)",
            "avatar_url": "VARCHAR(500) DEFAULT ''",
            "avatar_public_id": "VARCHAR(200)",
            "is_active": "BOOLEAN DEFAULT 1",
            "is_available": "BOOLEAN DEFAULT 1",
            "availability_note": "VARCHAR(250) DEFAULT ''",
        },
        "scans": {
            "lab_tech_id": "INTEGER REFERENCES users(id)",
        },
        "reports": {
            "reviewed_by_doctor_id": "INTEGER REFERENCES users(id)",
            "doctor_notes": "TEXT DEFAULT ''",
            "doctor_approved_at": "DATETIME",
            "forwarded_to_doctor_id": "INTEGER REFERENCES users(id)",
            "report_pdf_url": "VARCHAR(500)",
            "report_pdf_public_id": "VARCHAR(100)",
        },
        "pharmacy_inventory": {
            "expiry_date": "DATE",
        },
    }

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table_name, columns in additions.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            for column_name, definition in columns.items():
                if column_name not in existing_columns:
                    connection.execute(
                        text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {definition}')
                    )
