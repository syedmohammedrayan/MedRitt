"""
MedRittAI — Report Router
Retrieve LLM-generated reports and download as PDF.
"""

import json
import logging
import os
from types import SimpleNamespace
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from PIL import Image
from sqlalchemy.orm import Session

from config import settings
from db.database import get_db
from db import crud
from models.schemas import (
    PDFRequest,
    PatientSummaryRequest,
    PatientSummaryResponse,
    ReportData,
    ReportResponse,
    DoctorReviewRequest,
    ForwardReportRequest,
)
from routers.auth import get_current_user
from routers.workflow_utils import ensure_scan_access
from services.llm_report_engine import LLMReportEngine

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# GET REPORT
# ============================================================

@router.get("/{scan_id}", response_model=ReportResponse)
async def get_report(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Retrieve the generated clinical report for a scan.
    Returns the clinician-facing report data.
    """
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)

    # Get report from DB
    report = crud.get_report_by_scan(db, scan_id)
    if not report:
        if scan.status not in {"analyzing", "analyzed"}:
            raise HTTPException(
                status_code=404,
                detail="Report not yet generated. Run analysis first.",
            )
        raise HTTPException(
            status_code=404,
            detail="Report generation is still in progress. Please retry shortly.",
            headers={"Retry-After": "1"},
        )
    if current_user.role == "patient" and not report.doctor_approved_at:
        raise HTTPException(status_code=409, detail="Report is awaiting doctor approval")

    # Parse stored JSON
    try:
        report_data = json.loads(report.report_json)
    except json.JSONDecodeError:
        report_data = {}
    report_data = LLMReportEngine._ground_report_to_available_input(
        report_data,
        report_data.get("scan_type", scan.scan_type),
    )

    # Use edited versions if clinician has made edits
    findings = report.edited_findings or report_data.get("findings", "")
    impression = report.edited_impression or report_data.get("impression", "")

    result = crud.get_result_by_scan(db, scan_id)
    scan_orig_url = getattr(scan, "original_image_url", None) or f"/static/uploads/{scan_id}.png"
    scan_heatmap_url = getattr(scan, "heatmap_url", None) or f"/static/heatmaps/{scan_id}.png"
    result_overlay_url = getattr(result, "overlay_url", None) or scan_heatmap_url
    result_task_type = getattr(result, "task_type", None) or ("detection" if scan.scan_type in {"brain_tumor", "bone_fracture"} else "classification")
    raw_bboxes = getattr(result, "bounding_boxes", None)
    bboxes = []
    if raw_bboxes:
        try:
            bboxes = json.loads(raw_bboxes) if isinstance(raw_bboxes, str) else raw_bboxes
        except Exception:
            bboxes = []

    return ReportResponse(
        scan_id=scan_id,
        report=ReportData(
            patient_id=report.patient_id or "DEMO-001",
            scan_date=report_data.get("scan_date", ""),
            scan_type=report_data.get("scan_type", scan.scan_type),
            modality=report_data.get("modality", scan.modality),
            top_label=report_data.get("top_label", ""),
            confidence=report_data.get("confidence", 0.0),
            all_scores=report_data.get("all_scores", {}),
            clinical_history=report_data.get("clinical_history", "Not provided."),
            technique=report_data.get("technique", ""),
            image_quality=report_data.get("image_quality", ""),
            findings=findings,
            impression=impression,
            differential_diagnosis=report_data.get("differential_diagnosis", ""),
            recommendations=report_data.get("recommendations", ""),
            critical_communication=report_data.get("critical_communication", "No critical communication generated."),
            severity=report_data.get("severity", None),
            disclaimer=report_data.get("disclaimer", ""),
            generated_at=report.generated_at.isoformat() if report.generated_at else "",
            heatmap_target_label=report_data.get("heatmap_target_label") or "",
            is_low_confidence=bool(report_data.get("is_low_confidence", False)),
            methodology=report_data.get("methodology", ""),
            limitations=report_data.get("limitations", ""),
            doctor_assessment=report.doctor_notes or "",
            original_image_url=scan_orig_url,
            heatmap_url=scan_heatmap_url,
            overlay_url=result_overlay_url,
            task_type=result_task_type,
            bounding_boxes=bboxes,
        ),
    )


@router.post("/{scan_id}/regenerate", response_model=ReportResponse)
async def regenerate_report(
    scan_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Rebuild a stored clinical draft without rerunning image classification."""
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only an administrator can regenerate reports")
    stored_result = crud.get_result_by_scan(db, scan_id)
    if not stored_result:
        raise HTTPException(status_code=409, detail="Analyze the scan before generating a report")

    try:
        scores = json.loads(stored_result.all_scores or "{}")
    except json.JSONDecodeError:
        scores = {}
    sorted_secondary = [
        {"label": label, "score": float(score)}
        for label, score in sorted(scores.items(), key=lambda item: -float(item[1]))
        if label != stored_result.top_label and float(score) >= 0.20
    ][:3]
    try:
        bboxes = json.loads(stored_result.bounding_boxes or "[]")
    except json.JSONDecodeError:
        bboxes = []

    jeevansh_result = {
        "task_type": stored_result.task_type,
        "top_label": stored_result.top_label,
        "confidence": float(stored_result.confidence) if stored_result.confidence is not None else 0.0,
        "severity": stored_result.severity,
        "all_scores": scores,
        "detections": bboxes,
    }

    image_path = scan.file_path
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        image_path = os.path.join(settings.uploads_dir, f"{scan_id}.png")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Stored scan image is unavailable")

    try:
        with Image.open(image_path) as source:
            image = source.copy()

        # Phase 16 Orchestrated Report Generation
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is None:
            from services.ai_orchestrator import create_orchestrator
            orchestrator = create_orchestrator(request.app.state)
            request.app.state.orchestrator = orchestrator

        pipeline_result = await orchestrator.run_report_pipeline(
            jeevansh_result=jeevansh_result,
            scan_type=scan.scan_type,
            modality=scan.modality,
            image=image,
        )
        report_data = pipeline_result.report_data

        crud.replace_report(
            db=db,
            scan_id=scan_id,
            report_data=report_data,
            llm_provider=pipeline_result.llm_provider,
        )
        logger.info("Clinical report regenerated for %s", scan_id[:8])
        return await get_report(scan_id=scan_id, db=db, current_user=current_user)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report regeneration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not regenerate the clinical report")


# ============================================================
# DOWNLOAD PDF
# ============================================================

@router.post("/{scan_id}/pdf")
async def download_pdf(
    scan_id: str,
    request: Request,
    pdf_request: PDFRequest = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Generate and download a PDF report.

    Generated clinical sections are read-only; this endpoint exports the stored report.
    Returns PDF binary with Content-Disposition: attachment for auto-download.
    """
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)

    # Get report from DB
    report = crud.get_report_by_scan(db, scan_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated")
    if current_user.role == "patient" and not report.doctor_approved_at:
        raise HTTPException(status_code=409, detail="Report is awaiting doctor approval")

    try:
        report_data = json.loads(report.report_json)
    except json.JSONDecodeError:
        report_data = {}
    report_data = LLMReportEngine._ground_report_to_available_input(
        report_data,
        report_data.get("scan_type", scan.scan_type),
    )
    report_data["doctor_assessment"] = report.doctor_notes or ""
    report_data["is_final"] = bool(report.doctor_approved_at)

    # Check if we have a cached Cloudinary PDF and no edits are requested in this call
    # Note: pdf_request might contain new edits we want to apply to the PDF.
    has_new_edits = False
    if pdf_request and (pdf_request.edited_findings or pdf_request.edited_impression):
        has_new_edits = True

    # Generate the stored report with both Grad-CAM and original scan images
    pdf_generator = request.app.state.pdf_generator

    try:
        # Also check DB for previously saved edits
        if report.edited_findings:
            report_data["findings"] = report.edited_findings
        if report.edited_impression:
            report_data["impression"] = report.edited_impression

        # Resolve heatmap path for PDF embedding
        heatmap_path = os.path.join(settings.heatmaps_dir, f"{scan_id}.png")
        downloaded = False
        if not os.path.exists(heatmap_path):
            cloud_url = scan.heatmap_url
            if not cloud_url and getattr(scan, "result", None) and scan.result.overlay_url:
                cloud_url = scan.result.overlay_url

            if cloud_url and cloud_url.startswith("http"):
                try:
                    import requests
                    resp = requests.get(cloud_url, timeout=10)
                    if resp.status_code == 200:
                        with open(heatmap_path, "wb") as temp_f:
                            temp_f.write(resp.content)
                        downloaded = True
                except Exception as e:
                    logger.error(f"Failed to download heatmap from Cloudinary: {e}")

        if not os.path.exists(heatmap_path):
            heatmap_path = ""

        # Resolve original scan path for PDF embedding
        original_image_path = getattr(scan, "file_path", None)
        if not original_image_path or not os.path.exists(original_image_path):
            original_image_path = os.path.join(settings.uploads_dir, f"{scan_id}.png")

        downloaded_orig = False
        if not os.path.exists(original_image_path):
            cloud_orig_url = getattr(scan, "original_image_url", None)
            if cloud_orig_url and cloud_orig_url.startswith("http"):
                try:
                    import requests
                    resp = requests.get(cloud_orig_url, timeout=10)
                    if resp.status_code == 200:
                        temp_orig = os.path.join(settings.uploads_dir, f"{scan_id}_orig.png")
                        with open(temp_orig, "wb") as temp_f:
                            temp_f.write(resp.content)
                        original_image_path = temp_orig
                        downloaded_orig = True
                except Exception as e:
                    logger.error(f"Failed to download original scan from Cloudinary: {e}")

        if not os.path.exists(original_image_path):
            original_image_path = ""

        try:
            pdf_bytes = pdf_generator.generate_pdf(
                report_data=report_data,
                scan_id=scan_id,
                heatmap_path=heatmap_path,
                original_image_path=original_image_path,
            )
        finally:
            if downloaded and os.path.exists(heatmap_path):
                try:
                    os.remove(heatmap_path)
                except Exception:
                    pass
            if downloaded_orig and os.path.exists(original_image_path):
                try:
                    os.remove(original_image_path)
                except Exception:
                    pass

        filename = f"MedRittAI_Report_{scan_id[:8]}.pdf"

        logger.info(f"PDF generated for scan {scan_id[:8]} ({len(pdf_bytes)} bytes)")

        from services.cloudinary_storage import upload_report_pdf

        temp_pdf_path = os.path.join(settings.DATA_DIR, f"temp_{scan_id}.pdf")
        try:
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            pdf_url, pdf_pub_id = upload_report_pdf(temp_pdf_path)

            report.report_pdf_url = pdf_url
            report.report_pdf_public_id = pdf_pub_id
            db.commit()
            logger.info(f"PDF uploaded to Cloudinary: {pdf_url}")
        except Exception as e:
            logger.error(f"Cloudinary PDF upload failed: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}",
        )


@router.post("/{scan_id}/doctor-review", response_model=ReportResponse)
async def doctor_review(
    scan_id: str,
    payload: DoctorReviewRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Add the clinician assessment and optionally sign off the AI draft."""
    if current_user.role not in {"doctor", "admin"}:
        raise HTTPException(status_code=403, detail="Only doctors can review reports")
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)
    report = crud.get_report_by_scan(db, scan_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated")
    report.doctor_notes = payload.doctor_notes
    report.reviewed_by_doctor_id = current_user.id
    report.doctor_approved_at = datetime.now(timezone.utc) if payload.approve else None
    if scan.diagnostic_order and payload.approve:
        scan.diagnostic_order.status = "reviewed"
    db.commit()
    db.refresh(report)
    return await get_report(scan_id=scan_id, db=db, current_user=current_user)


@router.post("/{scan_id}/forward", response_model=ReportResponse)
async def forward_report(
    scan_id: str,
    payload: ForwardReportRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Forward a report into another specialist's review queue."""
    if current_user.role not in {"doctor", "admin"}:
        raise HTTPException(status_code=403, detail="Only doctors can forward reports")
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)
    target = crud.get_user(db, payload.doctor_id)
    if not target or target.role != "doctor" or not target.is_active:
        raise HTTPException(status_code=404, detail="Target doctor not found")
    report = crud.get_report_by_scan(db, scan_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated")
    report.forwarded_to_doctor_id = target.id
    db.commit()
    return await get_report(scan_id=scan_id, db=db, current_user=current_user)


# ============================================================
# PATIENT-FRIENDLY SUMMARY
# ============================================================

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Tamil", "Telugu", "Marathi",
    "Bengali", "Kannada", "Gujarati", "Malayalam", "Punjabi", "Urdu",
]


@router.post("/{scan_id}/patient-summary", response_model=PatientSummaryResponse)
async def get_patient_summary(
    scan_id: str,
    body: PatientSummaryRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Generate a patient-friendly summary of the medical report.

    Takes a language parameter and returns a simplified, non-technical
    explanation that patients can understand in their native language.
    """
    # Validate language
    language = body.language.strip().title()
    if language not in SUPPORTED_LANGUAGES:
        language = "English"

    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    ensure_scan_access(current_user, scan)
    if current_user.role == "patient" and not (
        scan.report and scan.report.doctor_approved_at
    ):
        raise HTTPException(status_code=409, detail="Report is awaiting doctor approval")

    # Get report from DB
    report = crud.get_report_by_scan(db, scan_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not yet generated")

    try:
        report_data = json.loads(report.report_json)
    except json.JSONDecodeError:
        report_data = {}
    report_data = LLMReportEngine._ground_report_to_available_input(
        report_data,
        report_data.get("scan_type", scan.scan_type),
    )

    if report.edited_findings:
        report_data["findings"] = report.edited_findings
    if report.edited_impression:
        report_data["impression"] = report.edited_impression

    # Generate patient summary
    report_engine = request.app.state.report_engine
    try:
        summary = await report_engine.generate_patient_report(
            report_data=report_data,
            language=language,
        )

        logger.info(f"Patient summary generated for {scan_id[:8]} in {language}")

        return {
            "scan_id": scan_id,
            "language": language,
            "summary": summary,
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    except Exception as e:
        logger.error(f"Patient summary failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Could not generate patient summary. Please try again.",
        )
