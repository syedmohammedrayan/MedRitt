"""
MedRittAI — Scan Upload & Analysis Router
Handles image upload, validation, and dual-model AI inference pipeline.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from config import settings
from db.database import get_db, get_session_factory
from db import crud
from models.schemas import UploadResponse, AnalysisResponse, ClassificationDetail, LocalizationDetail
from routers.auth import get_current_user
from routers.workflow_utils import ensure_scan_access
from services.scan_type_verifier import ScanTypeVerification

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "application/dicom", "application/octet-stream",
}


def _validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    # Check content type
    content_type = file.content_type or ""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in settings.ALLOWED_EXTENSIONS and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Accepted: PNG, JPEG, DICOM (.dcm)",
        )


def _validate_magic_bytes(file_bytes: bytes, filename: str) -> None:
    """Validate common image signatures before decoding."""
    ext = os.path.splitext(filename)[1].lower()
    is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
    is_dicom = (
        ext == ".dcm"
        and (len(file_bytes) > 132 and file_bytes[128:132] == b"DICM")
    )

    if not (is_png or is_jpeg or is_dicom):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported or invalid file content. Upload a valid PNG, JPEG, or DICOM image.",
        )


def _detect_modality(scan_type: str, filename: str) -> str:
    """Detect imaging modality from scan type and filename."""
    if scan_type == "brain_tumor":
        return "MRI"
    if scan_type == "pneumonia":
        return "X-ray"
    if scan_type == "bone_fracture":
        return "X-ray"
    if scan_type == "skin_cancer":
        return "Dermatoscopy"
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".dcm":
        return "DICOM"
    return "X-ray"


def _reject_bad_scan(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _normalized_grayscale(image: Image.Image, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    gray = np.asarray(image.resize(size).convert("L"), dtype=np.float32) / 255.0
    lo, hi = np.quantile(gray, [0.01, 0.99])
    if hi > lo:
        gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return gray


def _looks_like_centered_brain_slice(gray: np.ndarray) -> bool:
    """Detect common axial/sagittal MRI-like centered oval brain images."""
    mask = (gray > max(0.12, float(np.quantile(gray, 0.30)))).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return False

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    area = float(stats[largest, cv2.CC_STAT_AREA])

    image_area = float(gray.shape[0] * gray.shape[1])
    area_ratio = area / image_area
    bbox_fill = area / max(float(w * h), 1.0)
    aspect = w / max(float(h), 1.0)
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    centered = abs(center_x - gray.shape[1] / 2.0) < gray.shape[1] * 0.15 and abs(center_y - gray.shape[0] / 2.0) < gray.shape[0] * 0.18

    border = np.concatenate([gray[:16, :].ravel(), gray[-16:, :].ravel(), gray[:, :16].ravel(), gray[:, -16:].ravel()])
    dark_border = float(border.mean()) < 0.12

    return (
        centered
        and dark_border
        and 0.18 <= area_ratio <= 0.68
        and 0.65 <= aspect <= 1.45
        and bbox_fill >= 0.42
    )


def _has_chest_xray_lung_pattern(gray: np.ndarray) -> bool:
    """
    Check for coarse PA/AP chest projection structure:
    paired darker lung regions with a brighter central mediastinal column.
    """
    body_mask = gray > max(0.08, float(np.quantile(gray, 0.18)))
    body_ratio = float(body_mask.mean())
    if body_ratio < 0.42:
        return False

    left_lung = gray[70:190, 34:108]
    right_lung = gray[70:190, 148:222]
    center = gray[65:195, 110:146]
    upper_center = gray[35:95, 105:151]
    lower_center = gray[150:225, 82:174]

    lung_mean = float((left_lung.mean() + right_lung.mean()) / 2.0)
    center_mean = float(center.mean())
    upper_center_mean = float(upper_center.mean())
    lower_center_mean = float(lower_center.mean())

    dark_threshold = float(np.quantile(gray[body_mask], 0.38)) if body_mask.any() else 0.38
    left_dark = float((left_lung < dark_threshold).mean())
    right_dark = float((right_lung < dark_threshold).mean())
    paired_lungs = left_dark > 0.18 and right_dark > 0.18 and min(left_dark, right_dark) / max(left_dark, right_dark) > 0.35
    mediastinum_brighter = center_mean > lung_mean + 0.025 or upper_center_mean > lung_mean + 0.02
    lower_not_empty = lower_center_mean > lung_mean - 0.08

    return paired_lungs and mediastinum_brighter and lower_not_empty


def _validate_scan_matches_selected_type(image: Image.Image, scan_type: str, modality: str) -> None:
    """
    Validate basic image quality and authoritative DICOM modality metadata.

    Anatomy/modality verification for PNG/JPEG and body-part verification for
    DICOM are handled by the independent verifier after this cheap local gate.
    Fixed-region heuristics remain diagnostic-only because they reject valid
    projections and accept some wrong-modality images.
    """
    width, height = image.size
    if min(width, height) < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image is too small for reliable analysis. Upload a higher-resolution medical image.",
        )

    modality_upper = (modality or "").upper()

    # DICOM modality tags are authoritative. Values synthesized from the
    # selected type for PNG/JPEG files naturally agree with this branch.
    if scan_type in ["pneumonia", "bone_fracture"] and modality_upper in {"MR", "MRI", "CT", "US", "NM", "PT"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selected X-Ray based scan, but uploaded DICOM modality is {modality}. Select the correct scan type.",
        )
    if scan_type == "brain_tumor" and modality_upper in {"CR", "DX", "DR", "XR", "RG"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selected Brain Tumor (MRI), but uploaded DICOM modality is {modality}. Select the correct scan type.",
        )

    arr = np.asarray(image.resize((256, 256)).convert("RGB"), dtype=np.float32) / 255.0
    channel_delta = float(
        np.mean(
            np.abs(arr[:, :, 0] - arr[:, :, 1])
            + np.abs(arr[:, :, 1] - arr[:, :, 2])
            + np.abs(arr[:, :, 0] - arr[:, :, 2])
        )
        / 3.0
    )
    gray_rgb = arr.mean(axis=2)
    contrast = float(gray_rgb.std())
    dynamic_range = float(np.quantile(gray_rgb, 0.95) - np.quantile(gray_rgb, 0.05))

    # Reject only inputs that are clearly unusable by either grayscale model.
    # The relaxed colour threshold tolerates scanner overlays and compression.
    if channel_delta > 0.14:
        _reject_bad_scan(
            "This does not look like a grayscale medical image. "
            "Upload the original chest X-ray or brain MRI image."
        )
    if contrast < 0.02 or dynamic_range < 0.07:
        _reject_bad_scan(
            "This image does not have enough contrast for reliable analysis. "
            "Upload the original diagnostic image rather than a blank or heavily compressed preview."
        )

    # These signals are useful for observability, but are not reliable enough
    # to reject a selected modality without a trained OOD/modality classifier.
    gray = _normalized_grayscale(image)
    logger.debug(
        "Upload anatomy signals for selected %s: brain_like=%s chest_like=%s",
        scan_type,
        _looks_like_centered_brain_slice(gray),
        _has_chest_xray_lung_pattern(gray),
    )


def _enforce_scan_type_verification(
    verification: ScanTypeVerification,
    selected_scan_type: str,
    min_confidence: float,
) -> None:
    """Reject anything except a confident, single-image exact type match."""
    selected = "Brain MRI" if selected_scan_type == "brain_mri" else "Chest X-Ray"
    required = (
        "one original brain MRI slice showing intracranial anatomy"
        if selected_scan_type == "brain_mri"
        else "one original frontal chest X-ray showing the full thorax and both lungs"
    )
    if not verification.is_single_diagnostic_image:
        _reject_bad_scan(
            f"This is not an acceptable {selected} image. Upload {required}. "
            "Screenshots, posters, collages, report pages, and images dominated by text or interface elements are rejected."
        )
    if not verification.anatomy_complete_enough:
        _reject_bad_scan(
            f"This {selected} image is too cropped, obscured, or low quality. Upload {required}."
        )
    if verification.category == "other":
        _reject_bad_scan(
            f"This is not a valid {selected} image. Upload {required}; no analysis was performed."
        )
    if verification.category == "uncertain" or verification.confidence < min_confidence:
        _reject_bad_scan(
            f"We could not verify this as a valid {selected} image. Upload {required}; no analysis was performed."
        )
    if verification.category != selected_scan_type:
        detected = "brain MRI" if verification.category == "brain_mri" else "chest X-ray"
        _reject_bad_scan(
            f"The uploaded image appears to be a {detected}, but {selected} is selected. "
            "Choose the matching scan type before continuing."
        )


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_scan(
    request: Request,
    file: UploadFile = File(...),
    scan_type: str = Form(default="chest_xray"),
    diagnostic_order_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Upload a medical image for analysis.

    Args:
        file: Image file (PNG/JPEG/DICOM)
        scan_type: chest_xray, brain_mri, lung_ct, or kidney_us
    """
    if getattr(current_user, "role", "doctor") not in {"doctor", "lab_tech", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors and lab technicians can upload diagnostic scans",
        )
    _validate_file(file)

    # Validate scan_type
    supported_scan_types = ("skin_cancer", "pneumonia", "brain_tumor", "bone_fracture")
    if scan_type not in supported_scan_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"scan_type must be one of: {', '.join(supported_scan_types)}",
        )

    diagnostic_order = None
    if diagnostic_order_id is not None:
        diagnostic_order = crud.get_diagnostic_order(db, diagnostic_order_id)
        if not diagnostic_order:
            raise HTTPException(status_code=404, detail="Diagnostic order not found")
        if diagnostic_order.scan_type != scan_type:
            raise HTTPException(status_code=400, detail="Scan type does not match the diagnostic order")
        if diagnostic_order.scan_id:
            raise HTTPException(status_code=409, detail="A scan is already linked to this order")
        if current_user.role == "lab_tech":
            if diagnostic_order.assigned_lab_tech_id not in {None, current_user.id}:
                raise HTTPException(status_code=403, detail="Order is assigned to another technician")
            if diagnostic_order.assigned_lab_tech_id is None:
                crud.assign_lab_tech(db, diagnostic_order, current_user.id)
        elif current_user.role != "admin" and diagnostic_order.ordering_doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

    # Read file bytes
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB",
        )
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Generate scan ID
    scan_id = str(uuid.uuid4())
    filename = file.filename or f"scan_{scan_id[:8]}.png"
    _validate_magic_bytes(file_bytes, filename)
    modality = _detect_modality(scan_type, filename)

    # Process image
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".dcm":
            from services.dicom_parser import parse_dicom
            image, dicom_meta = parse_dicom(file_bytes)
            modality = dicom_meta.get("modality", modality)
        else:
            from io import BytesIO
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read image file: {str(e)}",
        )

    _validate_scan_matches_selected_type(image, scan_type, modality)


    # Save original image as PNG
    file_path = os.path.join(settings.uploads_dir, f"{scan_id}.png")
    image.save(file_path, "PNG")

    # Generate thumbnail (128×128)
    thumbnail = image.copy()
    thumbnail.thumbnail((128, 128), Image.Resampling.LANCZOS)
    thumbnail_path = os.path.join(settings.thumbnails_dir, f"{scan_id}.png")
    thumbnail.save(thumbnail_path, "PNG")


    # Cloudinary Upload
    from services.cloudinary_storage import upload_original_scan
    orig_url, orig_pub_id = None, None
    try:
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        orig_url, orig_pub_id = upload_original_scan(buffer)
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {e}")
        # Clean up local files
        if os.path.exists(file_path): os.remove(file_path)
        if os.path.exists(thumbnail_path): os.remove(thumbnail_path)
        raise HTTPException(status_code=500, detail="Failed to upload image to cloud storage.")

    # Save to database
    scan = crud.create_scan(
        db=db,
        scan_id=scan_id,
        user_id=(diagnostic_order.appointment.patient_id if diagnostic_order else current_user.id),
        filename=filename,
        scan_type=scan_type,
        modality=modality,
        file_path=file_path,
        thumbnail_path=thumbnail_path,
        file_size_bytes=file_size,
        original_image_url=orig_url,
        original_image_public_id=orig_pub_id,
        lab_tech_id=current_user.id if current_user.role == "lab_tech" else None,
    )
    if diagnostic_order:
        crud.link_scan_to_order(db, diagnostic_order, scan)

    logger.info(f"Scan uploaded: {scan_id[:8]} ({scan_type}, {file_size} bytes)")

    return UploadResponse(
        scan_id=scan_id,
        filename=filename,
        scan_type=scan_type,
        modality=modality,
        file_size_bytes=file_size,
        status="uploaded",
        uploaded_at=scan.uploaded_at.isoformat() if scan.uploaded_at else "",
        thumbnail_url=orig_url if orig_url else f"/static/thumbnails/{scan_id}.png",
    )


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@router.post("/analyze/{scan_id}", response_model=AnalysisResponse)
async def analyze_scan(
    scan_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Trigger AI inference on an uploaded scan.
    Runs classification → model attribution → severity → report generation.
    """
    # Get scan from DB
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID {scan_id[:8]} not found",
        )

    ensure_scan_access(current_user, scan)
    if getattr(current_user, "role", "doctor") not in {"doctor", "lab_tech", "admin"}:
        raise HTTPException(status_code=403, detail="Only clinical staff can run analysis")

    # Update status
    crud.update_scan_status(db, scan_id, "analyzing")

    start_time = time.time()

    report_task = None

    try:
        # Load image
        if os.path.exists(scan.file_path):
            image = Image.open(scan.file_path).convert("RGB")
        elif scan.original_image_url:
            import requests
            from io import BytesIO
            resp = requests.get(scan.original_image_url, timeout=10)
            if resp.status_code == 200:
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                raise RuntimeError(f"Could not download scan from Cloudinary. Status: {resp.status_code}")
        else:
            raise RuntimeError("Scan file not found locally or on Cloudinary")

        registry = getattr(request.app.state, "jeevansh_registry", None)
        if not registry:
            raise RuntimeError("Jeevansh model registry is not loaded.")
        model = registry.get_model(scan.scan_type)
        if not model:
            raise RuntimeError(f"Model not found in registry for {scan.scan_type}")

        # Run Jeevansh inference off event loop
        inference_result = await asyncio.to_thread(model.predict, image)
        task_type = inference_result.get("task_type", "classification")

        # Extract arrays for orchestrator
        heatmap_overlay = inference_result.get("gradcam") if task_type == "classification" else None
        overlay_array = inference_result.get("overlay") if task_type == "detection" else None

        # Execute Phase 16 Scan Orchestration
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is None:
            from services.ai_orchestrator import create_orchestrator
            orchestrator = create_orchestrator(request.app.state)
            request.app.state.orchestrator = orchestrator

        orchestrator_result = await orchestrator.run_scan_pipeline(
            image=image,
            scan_type=scan.scan_type,
            scan_id=scan_id,
            jeevansh_result=inference_result,
            heatmap_array=heatmap_overlay,
            overlay_array=overlay_array,
        )

        db_top_label = orchestrator_result.top_label or "Unknown"
        db_confidence = orchestrator_result.confidence or 0.0
        db_severity = orchestrator_result.severity
        db_all_scores = orchestrator_result.all_scores
        db_bboxes = orchestrator_result.bounding_boxes

        heatmap_url = orchestrator_result.heatmap_url
        heatmap_pub_id = orchestrator_result.heatmap_public_id
        overlay_url = orchestrator_result.overlay_url
        overlay_pub_id = orchestrator_result.overlay_public_id

        loc_type = "heatmap" if task_type == "classification" else "bbox"

        # We also need heatmap_overlay for the local static save
        heatmap_overlay_to_save = heatmap_overlay if task_type == "classification" else overlay_array

        # Ensure local original scan is saved in static uploads dir
        try:
            os.makedirs(settings.uploads_dir, exist_ok=True)
            local_upload_path = os.path.join(settings.uploads_dir, f"{scan_id}.png")
            if not os.path.exists(local_upload_path) and image is not None:
                image.save(local_upload_path, "PNG")
        except Exception as upload_save_err:
            logger.warning(f"Could not ensure local scan image for {scan_id[:8]}: {upload_save_err}")

        # Save overlay/heatmap locally so local static URL always works
        if heatmap_overlay_to_save is not None:
            try:
                os.makedirs(settings.heatmaps_dir, exist_ok=True)
                local_heatmap_path = os.path.join(settings.heatmaps_dir, f"{scan_id}.png")
                if isinstance(heatmap_overlay_to_save, np.ndarray):
                    if heatmap_overlay_to_save.dtype != np.uint8:
                        if heatmap_overlay_to_save.max() <= 1.0:
                            arr_to_save = (heatmap_overlay_to_save * 255).astype(np.uint8)
                        else:
                            arr_to_save = heatmap_overlay_to_save.astype(np.uint8)
                    else:
                        arr_to_save = heatmap_overlay_to_save
                    Image.fromarray(arr_to_save).save(local_heatmap_path, "PNG")
                elif hasattr(heatmap_overlay_to_save, "save"):
                    heatmap_overlay_to_save.save(local_heatmap_path, "PNG")

                if not overlay_url:
                    overlay_url = f"/static/heatmaps/{scan_id}.png"
                if not heatmap_url:
                    heatmap_url = f"/static/heatmaps/{scan_id}.png"
            except Exception as save_err:
                logger.warning(f"Could not save local overlay for {scan_id[:8]}: {save_err}")

        # Calculate analysis time
        analysis_time_ms = int((time.time() - start_time) * 1000)

        # Update scan with heatmap info
        crud.update_scan_heatmap(db, scan_id, None, heatmap_url=heatmap_url, heatmap_public_id=heatmap_pub_id)
        crud.update_scan_status(db, scan_id, "analyzed")

        # Store result in DB
        crud.create_result(
            db=db,
            scan_id=scan_id,
            task_type=task_type,
            model_id=inference_result.get("model_id"),
            top_label=db_top_label,
            confidence=db_confidence,
            severity=db_severity,
            all_scores=db_all_scores,
            bounding_boxes=db_bboxes,
            image_width=inference_result.get("image_width"),
            image_height=inference_result.get("image_height"),
            overlay_path=None,
            overlay_url=overlay_url,
            overlay_public_id=overlay_pub_id,
            analysis_time_ms=analysis_time_ms,
        )
        crud.complete_order_for_scan(db, scan_id)

        # Start the report task (Phase 16 Orchestrated)
        report_task = asyncio.create_task(
            orchestrator.run_report_pipeline(
                jeevansh_result=inference_result,
                scan_type=scan.scan_type,
                modality=scan.modality,
                image=image,
                patient_id=str(scan.user_id),
            ),
            name=f"report-{scan_id[:8]}",
        )

        # Do not hold the analysis response open for the report
        background_tasks.add_task(
            _store_generated_report,
            report_task,
            scan_id,
        )

        logger.info(
            f"Analysis complete: {scan_id[:8]} → {db_top_label} "
            f"({db_confidence * 100:.1f}%) in {analysis_time_ms}ms"
        )

        return AnalysisResponse(
            scan_id=scan_id,
            scan_type=scan.scan_type,
            original_image_url=scan.original_image_url or f"/static/uploads/{scan_id}.png",
            task_type=task_type,
            model_id=scan.scan_type,
            status="analyzed",
            classification=ClassificationDetail(
                top_label=db_top_label,
                confidence=db_confidence,
                severity=db_severity,
                all_scores=db_all_scores,
            ),
            localization=LocalizationDetail(
                type=loc_type,
                heatmap_url=heatmap_url or f"/static/heatmaps/{scan_id}.png",
                overlay_url=overlay_url or f"/static/heatmaps/{scan_id}.png",
                image_width=image.width,
                image_height=image.height,
                bounding_boxes=db_bboxes,
            ),
            analysis_time_ms=analysis_time_ms,
            analyzed_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        # Clean up temporary local files on failure
        try:
            if scan and os.path.exists(scan.file_path):
                os.remove(scan.file_path)
            if scan and scan.thumbnail_path and os.path.exists(scan.thumbnail_path):
                os.remove(scan.thumbnail_path)
        except Exception as ex:
            pass

        if report_task is not None and not report_task.done():
            report_task.cancel()
        crud.update_scan_status(db, scan_id, "failed")
        logger.error(f"Analysis failed for {scan_id[:8]}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


def _classify_chest_xray(request: Request, image: Image.Image):
    """Run chest X-ray classification."""
    return request.app.state.chest_classifier.predict(image)


def _localize_chest_xray(request: Request, image: Image.Image, result):
    """Generate the reference chest X-ray heatmap."""
    classifier = request.app.state.chest_classifier
    gradcam = request.app.state.chest_gradcam
    input_tensor = classifier.preprocess(image)
    heatmap_target_idx = result.heatmap_target_idx
    heatmap_target_label = result.heatmap_target_label
    heatmap_overlay = gradcam.generate_heatmap(
        image, input_tensor, heatmap_target_idx, target_label=heatmap_target_label,
    )
    raw_cam = gradcam.generate_raw_cam(input_tensor, heatmap_target_idx, image=image)
    bboxes = gradcam.heatmap_to_bboxes(raw_cam, threshold=0.6)
    return heatmap_overlay, bboxes


def _classify_brain_mri(request: Request, image: Image.Image):
    """Run brain MRI classification."""
    classifier = request.app.state.brain_classifier
    return classifier.predict(image)


def _localize_brain_mri(request: Request, image: Image.Image, result):
    """Generate brain MRI Grad-CAM++ localization for a classification."""
    classifier = request.app.state.brain_classifier
    gradcam = request.app.state.brain_gradcam

    # Generate Grad-CAM heatmap
    preprocessed = classifier.preprocess(image)
    # Generate the overlay and localization map together so the expensive
    # multi-layer gradient calculation runs only once.
    heatmap_overlay, raw_cam = gradcam.generate_heatmap_and_raw(image, preprocessed)

    # Extract bounding boxes
    bboxes = gradcam.heatmap_to_bboxes(raw_cam, threshold=0.5)

    return heatmap_overlay, bboxes


def _classify_lung_ct(request: Request, image: Image.Image):
    classifier = getattr(request.app.state, "lung_classifier", None)
    if classifier is None:
        raise RuntimeError("Lung CT model is unavailable")
    return classifier.predict(image)


def _localize_lung_ct(request: Request, image: Image.Image, result):
    return request.app.state.lung_gradcam.generate(
        image,
        result.heatmap_target_idx,
        result.heatmap_target_label,
    )


def _classify_kidney_us(request: Request, image: Image.Image):
    classifier = getattr(request.app.state, "kidney_classifier", None)
    if classifier is None:
        raise RuntimeError("Kidney ultrasound model is unavailable")
    return classifier.predict(image)


def _localize_kidney_us(request: Request, image: Image.Image, result):
    return request.app.state.kidney_gradcam.generate(
        image,
        result.heatmap_target_idx,
        result.heatmap_target_label,
    )


async def _store_generated_report(report_task, scan_id: str) -> None:
    """Persist a completed report using a session independent of the request."""
    started = time.perf_counter()
    try:
        pipeline_result = await report_task
        report_data = pipeline_result.report_data
        llm_provider = pipeline_result.llm_provider
        SessionLocal = get_session_factory()
        report_db = SessionLocal()
        try:
            if crud.get_scan(report_db, scan_id) is None:
                logger.info("Discarding report for deleted scan %s", scan_id[:8])
                return
            crud.replace_report(
                db=report_db,
                scan_id=scan_id,
                report_data=report_data,
                llm_provider=llm_provider,
            )
        finally:
            report_db.close()

        logger.info(
            "Clinical report ready for %s via %s (background store %.0fms)",
            scan_id[:8],
            report_data.get("llm_provider", "template"),
            (time.perf_counter() - started) * 1000,
        )
    except asyncio.CancelledError:
        logger.info("Clinical report generation cancelled for %s", scan_id[:8])
    except Exception:
        logger.exception("Clinical report generation failed for %s", scan_id[:8])


# Need datetime for the endpoint
from datetime import datetime
