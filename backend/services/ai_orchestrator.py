"""
MedRittAI Phase 16 — AI Orchestration Layer
============================================

Central coordinator for the four AI providers.  The orchestrator
determines which service is called, in which order, what data each
service receives, and how the final structured result is assembled and
persisted.

AUTHORITY MODEL (non-negotiable):
    Jeevansh  — authoritative medical diagnostic engine
    Gemini    — primary clinical report synthesis
    Groq      — fast multimodal scan-type verifier
    NVIDIA    — secondary report quality / grounding QA

NEVER allow Gemini, Groq, or NVIDIA to override Jeevansh diagnostic
truth.  The orchestrator itself MUST NOT manufacture diagnostic results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProviderStatus(str, Enum):
    SUCCESS     = "success"
    UNAVAILABLE = "unavailable"
    FAILED      = "failed"
    SKIPPED     = "skipped"
    BYPASSED    = "bypassed"


class ReportDecision(str, Enum):
    ACCEPTED          = "accepted"
    ACCEPTED_NO_QA    = "accepted_no_qa"
    FALLBACK_TEMPLATE = "fallback_template"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProviderMeta:
    status: ProviderStatus = ProviderStatus.SKIPPED
    detail: Optional[str] = None
    authoritative: bool = False


@dataclass
class OrchestrationContext:
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scan_id: Optional[str] = None
    scan_type: Optional[str] = None


@dataclass
class ScanPipelineResult:
    """Result of run_scan_pipeline(). Jeevansh fields are authoritative and immutable."""
    # Orchestration
    correlation_id: str = ""
    # Authoritative diagnostic (Jeevansh)
    task_type: Optional[str] = None
    top_label: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    all_scores: Optional[Dict] = None
    bounding_boxes: Optional[List] = None
    # Cloudinary asset references
    original_image_url: Optional[str] = None
    original_image_public_id: Optional[str] = None
    heatmap_url: Optional[str] = None
    heatmap_public_id: Optional[str] = None
    overlay_url: Optional[str] = None
    overlay_public_id: Optional[str] = None
    # Verification
    scan_type_verified: bool = True
    groq_verification_detail: Optional[str] = None
    # Provider provenance
    providers: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ReportPipelineResult:
    """Result of run_report_pipeline()."""
    correlation_id: str = ""
    report_data: Optional[Dict] = None
    llm_provider: str = "template"
    decision: ReportDecision = ReportDecision.FALLBACK_TEMPLATE
    nvidia_qa: Optional[Dict] = None
    providers: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MedRittOrchestrator:
    """
    Coordinates Jeevansh, Gemini, Groq, and NVIDIA.
    Calls existing services; does NOT duplicate their logic.
    """

    def __init__(
        self,
        *,
        report_engine,
        scan_verifier,
        nvidia_service=None,
        cloudinary_enabled: bool = True,
    ):
        self._report_engine = report_engine
        self._scan_verifier = scan_verifier
        self._nvidia_service = nvidia_service
        self._cloudinary_enabled = cloudinary_enabled

    async def run_scan_pipeline(
        self,
        *,
        image,
        scan_type: str,
        scan_id: str,
        jeevansh_result: dict,
        original_image_buffer=None,
        heatmap_array=None,
        overlay_array=None,
    ) -> ScanPipelineResult:
        ctx = OrchestrationContext(scan_id=scan_id, scan_type=scan_type)
        cid = ctx.correlation_id
        out = ScanPipelineResult(correlation_id=cid)
        providers: Dict[str, Dict] = {}

        logger.info("[%s] orchestration_started scan_id=%s scan_type=%s", cid, scan_id[:8], scan_type)

        # 1. Groq scan-type verification (via existing ScanTypeVerifier)
        groq_meta = ProviderMeta()
        try:
            verification = await asyncio.wait_for(
                self._scan_verifier.verify(image), timeout=35.0
            )
            groq_meta.status = ProviderStatus.SUCCESS
            out.scan_type_verified = verification.anatomy_complete_enough and verification.is_single_diagnostic_image
            out.groq_verification_detail = verification.reason
            logger.info("[%s] groq_verification_completed category=%s", cid, verification.category)
        except asyncio.TimeoutError:
            groq_meta.status = ProviderStatus.FAILED
            groq_meta.detail = "timeout"
            out.scan_type_verified = True
            logger.warning("[%s] Groq verification timed out — proceeding", cid)
        except Exception as exc:
            groq_meta.status = ProviderStatus.UNAVAILABLE
            groq_meta.detail = str(exc)
            out.scan_type_verified = True
            logger.warning("[%s] Groq verification unavailable: %s", cid, exc)
        providers["groq"] = {"status": groq_meta.status.value, "detail": groq_meta.detail}

        # 2. Lock in Jeevansh authoritative result (NEVER modified)
        task_type = jeevansh_result.get("task_type", "classification")
        out.task_type = task_type
        out.top_label = jeevansh_result.get("top_label")
        out.confidence = jeevansh_result.get("confidence")
        out.bounding_boxes = jeevansh_result.get("detections") or []
        if task_type == "classification":
            out.all_scores = jeevansh_result.get("all_scores")
            out.severity = jeevansh_result.get("severity")
        else:
            out.all_scores = None   # always null for detection
            out.severity = None     # never derive from detection confidence

        providers["jeevansh"] = {"status": ProviderStatus.SUCCESS.value, "authoritative": True}
        logger.info("[%s] jeevansh_completed task_type=%s top_label=%s", cid, task_type, out.top_label)

        # 3. Cloudinary uploads (isolated — failure does not abort pipeline)
        if self._cloudinary_enabled:
            if original_image_buffer is not None:
                try:
                    from services.cloudinary_storage import upload_original_scan
                    url, pub_id = upload_original_scan(original_image_buffer)
                    out.original_image_url = url
                    out.original_image_public_id = pub_id
                except Exception as exc:
                    logger.error("[%s] Cloudinary original scan upload failed: %s", cid, exc)

            if heatmap_array is not None and task_type == "classification":
                try:
                    from services.cloudinary_storage import upload_gradcam
                    url, pub_id = upload_gradcam(heatmap_array)
                    out.heatmap_url = url
                    out.heatmap_public_id = pub_id
                    out.overlay_url = url
                    out.overlay_public_id = pub_id
                except Exception as exc:
                    logger.error("[%s] Cloudinary Grad-CAM upload failed: %s", cid, exc)

            if overlay_array is not None and task_type == "detection":
                try:
                    from services.cloudinary_storage import upload_yolo_overlay
                    url, pub_id = upload_yolo_overlay(overlay_array)
                    out.overlay_url = url
                    out.overlay_public_id = pub_id
                except Exception as exc:
                    logger.error("[%s] Cloudinary YOLO overlay upload failed: %s", cid, exc)

        out.providers = providers
        logger.info("[%s] orchestration_completed scan_id=%s", cid, scan_id[:8])
        return out

    async def run_report_pipeline(
        self,
        *,
        jeevansh_result,
        scan_type: str,
        modality: str = "X-ray",
        patient_id: str = "P-",
        image=None,
    ) -> ReportPipelineResult:
        cid = str(uuid.uuid4())
        out = ReportPipelineResult(correlation_id=cid)
        providers: Dict[str, Dict] = {
            "jeevansh": {"status": ProviderStatus.SUCCESS.value, "authoritative": True}
        }

        logger.info("[%s] orchestration_started report scan_type=%s", cid, scan_type)

        # Gemini report synthesis via existing LLMReportEngine
        report_data: Optional[Dict] = None
        llm_provider = "template"
        gemini_meta = ProviderMeta()
        try:
            report_data = await asyncio.wait_for(
                self._report_engine.generate_report(
                    result=jeevansh_result,
                    scan_type=scan_type,
                    modality=modality,
                    patient_id=patient_id,
                    image=image,
                ),
                timeout=120.0,
            )
            llm_provider = report_data.get("llm_provider", "template")
            gemini_meta.status = (
                ProviderStatus.SUCCESS if llm_provider == "gemini" else ProviderStatus.BYPASSED
            )
            gemini_meta.detail = f"provider={llm_provider}"
            logger.info("[%s] gemini_report_completed provider=%s", cid, llm_provider)
        except asyncio.TimeoutError:
            gemini_meta.status = ProviderStatus.FAILED
            gemini_meta.detail = "timeout"
            logger.error("[%s] Gemini report timed out", cid)
        except Exception as exc:
            gemini_meta.status = ProviderStatus.FAILED
            gemini_meta.detail = str(exc)
            logger.error("[%s] Gemini report failed: %s", cid, exc)

        providers["gemini"] = {"status": gemini_meta.status.value, "detail": gemini_meta.detail}

        if report_data is None:
            try:
                report_data = self._report_engine._generate_template_report(jeevansh_result, scan_type)
                llm_provider = "template"
            except Exception as exc:
                out.error = f"Report generation failed entirely: {exc}"
                out.providers = providers
                return out

        # NVIDIA QA for LLM reports
        nvidia_meta = ProviderMeta()
        qa_result_dict: Optional[Dict] = None
        decision = ReportDecision.ACCEPTED_NO_QA

        if llm_provider in {"gemini", "groq", "nvidia"} and self._nvidia_service:
            diag_dict = {
                "top_label": getattr(jeevansh_result, "top_label", None),
                "confidence": getattr(jeevansh_result, "confidence", 0.0),
                "all_scores": getattr(jeevansh_result, "all_scores", {}),
                "task_type": getattr(jeevansh_result, "task_type", "classification"),
            }
            if getattr(jeevansh_result, "task_type", None) == "detection":
                diag_dict["bounding_boxes"] = getattr(jeevansh_result, "bounding_boxes", [])
            try:
                qa = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._nvidia_service.verify_report,
                        diag_dict,
                        json.dumps(report_data),
                    ),
                    timeout=65.0,
                )
                if qa is None:
                    nvidia_meta.status = ProviderStatus.UNAVAILABLE
                    decision = ReportDecision.ACCEPTED_NO_QA
                    logger.warning("[%s] NVIDIA unavailable — retaining %s report", cid, llm_provider)
                elif not qa.passes or qa.recommendation.lower() == "flag":
                    nvidia_meta.status = ProviderStatus.SUCCESS
                    nvidia_meta.detail = f"qa=flag"
                    qa_result_dict = qa.model_dump()
                    report_data = self._report_engine._generate_template_report(jeevansh_result, scan_type)
                    llm_provider = "template"
                    decision = ReportDecision.FALLBACK_TEMPLATE
                    logger.warning("[%s] nvidia_qa_completed qa=flag — template fallback", cid)
                else:
                    nvidia_meta.status = ProviderStatus.SUCCESS
                    qa_result_dict = qa.model_dump()
                    decision = ReportDecision.ACCEPTED
                    logger.info("[%s] nvidia_qa_completed qa=accept", cid)
            except asyncio.TimeoutError:
                nvidia_meta.status = ProviderStatus.FAILED
                nvidia_meta.detail = "timeout"
                decision = ReportDecision.ACCEPTED_NO_QA
                logger.warning("[%s] NVIDIA QA timed out — retaining %s report", cid, llm_provider)
            except Exception as exc:
                nvidia_meta.status = ProviderStatus.FAILED
                nvidia_meta.detail = str(exc)
                decision = ReportDecision.ACCEPTED_NO_QA
                logger.error("[%s] NVIDIA QA error: %s", cid, exc)
        else:
            nvidia_meta.status = ProviderStatus.SKIPPED
            nvidia_meta.detail = "not run — report is template"
            decision = ReportDecision.FALLBACK_TEMPLATE if llm_provider != "gemini" else ReportDecision.ACCEPTED_NO_QA

        providers["nvidia"] = {
            "status": nvidia_meta.status.value,
            "detail": nvidia_meta.detail,
            "qa_result": qa_result_dict,
        }

        out.report_data = report_data
        out.llm_provider = llm_provider
        out.decision = decision
        out.nvidia_qa = qa_result_dict
        out.providers = providers

        logger.info("[%s] report_persisted provider=%s decision=%s", cid, llm_provider, decision.value)
        logger.info("[%s] orchestration_completed report", cid)
        return out


def create_orchestrator(app_state) -> MedRittOrchestrator:
    """Factory — called once at startup, builds orchestrator from app.state services."""
    import os
    cloudinary_enabled = os.environ.get("CLOUDINARY_ENABLED", "false").lower() == "true"
    scan_verifier = getattr(app_state, "scan_verifier", None) or getattr(app_state, "scan_type_verifier", None)
    return MedRittOrchestrator(
        report_engine=getattr(app_state, "report_engine", None),
        scan_verifier=scan_verifier,
        nvidia_service=getattr(app_state, "nvidia_service", None),
        cloudinary_enabled=cloudinary_enabled,
    )
