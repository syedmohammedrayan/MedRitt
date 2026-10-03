"""
MedRittAI - Doctor Report Index Router
Read-only, authorization-scoped listing of diagnostic reports for the
doctor workspace "Reports" page.

Visibility is derived exclusively from the authenticated user (MedRitt JWT)
and database relationships:

    current doctor -> appointments.doctor_id -> diagnostic_orders -> scans -> reports
    current doctor -> diagnostic_orders.ordering_doctor_id -> scans -> reports
    reports.forwarded_to_doctor_id == current doctor

No patient_id / doctor_id / role is accepted from the client.
"""

import json
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from db.database import get_db
from db.models import Appointment, DiagnosticOrder, Report, Scan, User
from models.schemas import (
    DetectionSummary,
    DoctorReportIndexResponse,
    ReportSummary,
    ReportSummaryPatient,
)
from routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_SEARCH_LENGTH = 100


def _detection_summary(result) -> Optional[DetectionSummary]:
    if result is None or result.task_type != "detection":
        return None
    try:
        boxes = json.loads(result.bounding_boxes or "[]")
    except (json.JSONDecodeError, TypeError):
        boxes = []
    if not isinstance(boxes, list):
        boxes = []
    classes = sorted({str(box.get("class")) for box in boxes if isinstance(box, dict) and box.get("class")})
    return DetectionSummary(count=len(boxes), classes=classes)


def _summarize(report, scan, patient, current_user) -> ReportSummary:
    result = scan.result
    is_detection = result is not None and result.task_type == "detection"
    tested_at = scan.uploaded_at

    top_label = result.top_label if result is not None else None
    confidence = result.confidence if result is not None else None
    if is_detection and top_label == "No Detection":
        # Detection is not classification: no positive finding to report.
        top_label, confidence = None, None

    return ReportSummary(
        report_id=report.id,
        scan_id=scan.id,
        patient=ReportSummaryPatient(
            id=patient.id,
            username=patient.username,
            full_name=patient.full_name or patient.username,
            email=patient.email or "",
            phone=patient.phone or "",
        ),
        scan_type=scan.scan_type,
        modality=scan.modality or "",
        task_type=result.task_type if result is not None else None,
        tested_at=tested_at,
        test_date=tested_at.strftime("%Y-%m-%d") if tested_at else None,
        test_time=tested_at.strftime("%H:%M:%S") if tested_at else None,
        scan_status=scan.status or "",
        report_status="reviewed" if report.doctor_approved_at else "pending_review",
        top_label=top_label,
        confidence=confidence,
        severity=None if is_detection else (result.severity if result is not None else None),
        detection=_detection_summary(result),
        generated_at=report.generated_at,
        doctor_approved=report.doctor_approved_at is not None,
        doctor_approved_at=report.doctor_approved_at,
        reviewed_by_doctor_id=report.reviewed_by_doctor_id,
        forwarded_to_me=report.forwarded_to_doctor_id == current_user.id,
    )


@router.get("/doctor", response_model=DoctorReportIndexResponse)
def list_doctor_reports(
    search: Optional[str] = Query(default=None, max_length=MAX_SEARCH_LENGTH),
    scan_type: Optional[str] = Query(default=None, max_length=30),
    status: Optional[Literal["reviewed", "pending_review"]] = Query(default=None),
    sort: Literal["newest", "oldest"] = Query(default="newest"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if current_user.role not in {"doctor", "admin"}:
        raise HTTPException(status_code=403, detail="Only doctors can view the report index.")

    query = (
        db.query(Report, Scan, User)
        .join(Scan, Report.scan_id == Scan.id)
        .join(User, Scan.user_id == User.id)
        .outerjoin(DiagnosticOrder, DiagnosticOrder.scan_id == Scan.id)
        .outerjoin(Appointment, DiagnosticOrder.appointment_id == Appointment.id)
        .options(joinedload(Scan.result))
    )

    if current_user.role == "doctor":
        query = query.filter(
            or_(
                Appointment.doctor_id == current_user.id,
                DiagnosticOrder.ordering_doctor_id == current_user.id,
                Report.forwarded_to_doctor_id == current_user.id,
            )
        )
    else:
        # Admin: existing role rules grant full scan access; the index is
        # limited to patient reports (staff direct uploads are excluded).
        query = query.filter(User.role == "patient")

    term = (search or "").strip().lower()
    if term:
        like = f"%{term}%"
        query = query.filter(
            or_(
                func.lower(func.coalesce(User.full_name, "")).like(like),
                func.lower(User.username).like(like),
            )
        )

    if scan_type:
        query = query.filter(Scan.scan_type == scan_type)

    if current_user.role == "doctor" and current_user.department:
        dept_name = current_user.department.name
        dept_scan_types = {
            "Dermatology": ["skin_cancer"],
            "Neurology": ["brain_tumor"],
            "Pulmonology": ["pneumonia"],
            "Orthopedics": ["bone_fracture"],
        }
        if dept_name in dept_scan_types:
            query = query.filter(Scan.scan_type.in_(dept_scan_types[dept_name]))

    if status == "reviewed":
        query = query.filter(Report.doctor_approved_at.isnot(None))
    elif status == "pending_review":
        query = query.filter(Report.doctor_approved_at.is_(None))

    order_col = Scan.uploaded_at.desc() if sort == "newest" else Scan.uploaded_at.asc()
    rows = query.order_by(order_col, Report.id.desc()).all()

    seen = set()
    reports = []
    for report, scan, patient in rows:
        if report.id in seen:
            continue
        seen.add(report.id)
        reports.append(_summarize(report, scan, patient, current_user))

    return DoctorReportIndexResponse(reports=reports, total=len(reports))
