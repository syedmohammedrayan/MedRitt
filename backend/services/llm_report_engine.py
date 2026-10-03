"""
MedRittAI — LLM-Powered Clinical Report Engine
Generates professional radiology reports by sending model output to LLM APIs.

Flow: Image + Model Output → Grounded Clinical Prompt → Structured Report
Patient communication: English explanation → Sarvam translation → internal fallback
"""

import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# AI Disclaimer — mandatory on every report
DISCLAIMER = (
    "This preliminary report was prepared by the MedRittAI clinical support model "
    "and is intended as a decision-support tool only. "
    "It must NOT be used as a standalone diagnostic instrument. "
    "All findings must be reviewed, verified, and co-signed by a "
    "licensed radiologist before any clinical action is taken. "
    "MedRittAI is not a certified medical device."
)


SYSTEM_PROMPT = """You prepare a structured preliminary imaging report for a doctor.
Write with the organization, precision, and restraint of a careful senior radiologist.
You receive:
1. the uploaded medical image when a multimodal provider is available,
2. the local ML classifier output,
3. model confidence scores and severity metadata.

GROUNDING RULES:
- The image is the visual evidence. Classifier output is supporting evidence and must never be copied as a visual fact when the image does not support it.
- This may be a single exported image, not a complete imaging study. State that limitation in technique and image_quality.
- Never fabricate clinical history, symptoms, patient age/sex, projection, MRI sequence, contrast use, comparison, measurements, laterality, anatomical location, devices, or prior studies.
- Only state laterality, location, morphology, mass effect, edema, pleural findings, support devices, or measurements when clearly visible.
- Do not convert classifier confidence into clinical severity, urgency, tumor grade, or disease stage.
- If image and classifier disagree, say the examination is indeterminate and explain what confirmatory review is needed.
- Use concise radiology language. Do not discuss model architecture, provider names, prompts, confidence percentages, or heatmaps in the clinical prose.
- Do not mention the classifier, automated analysis, model agreement, concordance, confidence, Grad-CAM, or model attribution anywhere in the clinical sections. Those results are displayed separately.
- Findings must contain observations only, organized by anatomic system. Do not place diagnoses, differential weighting, or management advice in findings.
- Put diagnostic conclusions only in impression, ordered and numbered by clinical priority. Keep the separate differential section brief and consistent with that impression.
- When a finding can be measured from the available image, use a specific numeric measurement with units. Never substitute vague size language such as "large," "significant," or "substantial."
- For a brain MRI image with unknown sequence and contrast status, never describe enhancement, restricted diffusion, ADC, FLAIR, T1/T2 signal, susceptibility, perfusion, or contrast uptake as an observed finding.
- Put the most clinically important supported conclusion first in impression. Number multiple impressions.
- Differential diagnosis must be short and evidence-based. Use "None based on the supplied image" when no differential is supported.
- Recommendations must follow from the observed finding and limitations. Do not recommend biopsy, surgery, emergency treatment, or disease-specific laboratory testing solely from a classifier label.
- critical_communication must be "Not applicable — no critical or emergent finding requiring direct communication" unless the supplied image clearly demonstrates an immediately dangerous finding. For an urgent finding, document the recipient, method, date, and time when those details are available; never invent them.

You MUST output ONLY valid JSON with exactly these nine string keys:
{
  "technique": "What image/view was supplied and the limits of that input",
  "comparison": "Prior study availability",
  "image_quality": "Diagnostic adequacy and visible technical limitations",
  "findings": "Detailed, anatomy-organized observations supported by the image",
  "impression": "Numbered prioritized conclusions",
  "differential_diagnosis": "Brief supported differential or none",
  "recommendations": "Actionable next steps proportional to the evidence",
  "critical_communication": "Critical communication record, or the routine N/A statement",
  "patient_explanation": "Four short plain-English paragraphs explaining the supported conclusion, meaning, next step, and limitations without model terms or confidence scores"
}

Output ONLY valid JSON. No markdown, no code fences, no explanation, no extra text."""


class LLMReportEngine:
    """
    Generates clinical reports using LLM APIs.

    Tries providers in priority order:
    1. MAIRA-2 — chest X-ray report generation
    2. Gemini — image-aware multimodal reporting
    3. Groq (Llama 3.1 70B) — text-only fallback
    4. Anthropic Claude 3 Haiku — text-only fallback
    5. OpenAI GPT-4o-mini — text-only fallback
    6. Template fallback — no API key needed
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-3.8-flash",
        sarvam_api_key: Optional[str] = None,
        sarvam_translate_model: str = "sarvam-translate:v1",
        groq_api_key: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    ):
        self.gemini_key = gemini_api_key
        self.gemini_model = gemini_model

        self.sarvam_key = sarvam_api_key
        self.sarvam_translate_model = sarvam_translate_model
        self.sarvam_translation_disabled = False
        self.groq_key = groq_api_key

        from .gemini_service import GeminiService
        self.gemini_service = GeminiService(api_key=self.gemini_key, model=self.gemini_model) if self.gemini_key else None

        from .groq_service import GroqService
        self.groq_service = GroqService(api_key=self.groq_key) if self.groq_key else None

        from .nvidia_service import NVIDIAService
        self.nvidia_service = NVIDIAService(api_key=nvidia_api_key, model=nvidia_model) if nvidia_api_key else None

        providers = []
        if self.gemini_key:
            providers.append("gemini")
        if self.groq_key:
            providers.append("groq")
        if nvidia_api_key:
            providers.append("nvidia")
        if not providers:
            providers.append("template (fallback)")

        logger.info(
            "Clinical report engine initialized (%s); patient translation: %s",
            ", ".join(providers),
            "configured" if self.sarvam_key else "local fallback",
        )


    def close(self) -> None:
        """Release reusable network resources during application shutdown."""
        pass

    def _build_user_prompt(self, result, scan_type: str) -> str:
        """Build the clinical context prompt from model output."""
        if isinstance(result, dict):
            from types import SimpleNamespace
            result = SimpleNamespace(
                top_label=result.get("top_label") or "Diagnostic finding",
                confidence=float(result.get("confidence") or 0.0),
                all_scores=result.get("all_scores") or {},
                bounding_boxes=result.get("bounding_boxes") or result.get("detections") or [],
                task_type=result.get("task_type") or ("detection" if scan_type in {"brain_tumor", "bone_fracture"} else "classification"),
                severity=result.get("severity"),
            )
        if scan_type == "chest_xray":
            # Format all scores for context
            supported_scores = [
                (label, score)
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
                if score >= 0.20
            ][:5]
            scores_text = "\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in supported_scores
            ) or "  - No secondary score reached the reporting threshold."
            return f"""EXAM: Chest radiograph
AVAILABLE INPUT: One uploaded image only; projection and patient positioning are not provided.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING CLASSIFIER OUTPUT (not a substitute for visual findings):
- Highest-scoring label: {result.top_label} ({result.confidence * 100:.1f}%)
- Labels at or above the reporting threshold:
{scores_text}

CHEST-SPECIFIC INSTRUCTIONS:
- Organize findings under these plain-text labels: LUNGS/AIRWAYS, PLEURA, CARDIOMEDIASTINAL SILHOUETTE, HILA, BONES/SOFT TISSUES, SUPPORT DEVICES.
- Assess only what is visible. If a structure cannot be assessed, say so instead of assuming normality.
- Do not infer AP versus PA, portable technique, inspiration, rotation, or upright/supine position unless clearly demonstrated.
- "Normal" means that class scored highest among only three trained classes; it does not prove a normal radiograph.
- A positive classifier label may be reported as visually suspected only if the image supports it; otherwise describe the discordance in the impression.
- Do not include educational definitions of diseases in the report.

Generate a detailed, grounded chest radiograph report for clinical review."""

        elif scan_type == "brain_mri":
            # Format all 4-class scores for context
            scores_text = "\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
            )
            return f"""EXAM: Brain MRI image
AVAILABLE INPUT: One uploaded 2D image only; the sequence, plane, contrast status, and complete series are not provided.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING FOUR-CLASS CLASSIFIER OUTPUT (not a substitute for visual findings):
- Highest-scoring label: {result.top_label} ({result.confidence * 100:.1f}%)
- Class scores:
{scores_text}

CLASS DEFINITIONS:
- Glioma: Primary brain tumor arising from glial cells; includes astrocytomas, oligodendrogliomas, glioblastomas
- Meningioma: Typically benign tumor arising from meninges; usually well-circumscribed, extra-axial
- No Tumor: No model-supported mass lesion identified
- Pituitary: Tumor of the pituitary gland (sellar/suprasellar region); usually adenoma

BRAIN-SPECIFIC INSTRUCTIONS:
- Organize findings under these plain-text labels: BRAIN PARENCHYMA, EXTRA-AXIAL SPACES, VENTRICLES/MASS EFFECT, SELLA, VISIBLE POSTERIOR FOSSA.
- A single image cannot establish a complete brain MRI interpretation. Explicitly state which areas or features cannot be assessed.
- Do not infer MRI sequence, signal characteristics across sequences, enhancement, diffusion restriction, perfusion, hemorrhage, or exact tumor boundaries from absent series.
- Do not assign tumor grade, molecular subtype, histology, stage, or treatment urgency from the class label.
- "No Tumor" means no model-supported lesion in the four trained categories; it does not exclude other intracranial disease.
- Name a tumor category in the impression only when image appearance and classifier output are concordant. Otherwise use "indeterminate abnormality" and recommend complete diagnostic MRI review.
- Do not include general disease definitions in findings.

Generate a detailed, grounded limited-image neuroradiology report for clinical review."""

        elif scan_type in {"lung_ct", "kidney_us"}:
            exam_name = "Lung CT image" if scan_type == "lung_ct" else "Kidney ultrasound image"
            scores_text = "\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
            )
            organ_instructions = (
                "Describe only visible lung parenchyma and lesions. Do not infer staging, histology, metastasis, or treatment from the classifier category."
                if scan_type == "lung_ct"
                else "Describe only visible renal structures and echogenic foci. Do not infer obstruction, hydronephrosis, stone size, or laterality unless clearly shown."
            )
            return f"""EXAM: {exam_name}
AVAILABLE INPUT: One uploaded 2D image only; the complete study and acquisition metadata are not provided.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING CLASSIFIER OUTPUT (not a substitute for visual findings):
- Highest-scoring label: {result.top_label} ({result.confidence * 100:.1f}%)
- Class scores:
{scores_text}

ORGAN-SPECIFIC INSTRUCTIONS:
- {organ_instructions}
- Explicitly state the limitation of a single exported image.
- Do not convert confidence into stage, grade, urgency, or definitive diagnosis.
- Recommend review of the complete source study and clinical correlation.

Generate a detailed, grounded preliminary report for clinical review."""

        elif scan_type == "pneumonia":
            scores_text = "\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
                if score >= 0.05
            ) or "  - No secondary score reached the reporting threshold."
            return f"""EXAM: Chest radiograph
AVAILABLE INPUT: One uploaded image only; projection and patient positioning are not provided.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING CLASSIFIER OUTPUT (not a substitute for visual findings):
- Highest-scoring label: {result.top_label} ({result.confidence * 100:.1f}%)
- Labels at or above the reporting threshold:
{scores_text}

CHEST-SPECIFIC INSTRUCTIONS:
- Organize findings under these plain-text labels: LUNGS/AIRWAYS, PLEURA, CARDIOMEDIASTINAL SILHOUETTE, HILA, BONES/SOFT TISSUES, SUPPORT DEVICES.
- Assess only what is visible. If a structure cannot be assessed, say so instead of assuming normality.
- "NORMAL" means that class scored highest; it does not prove a normal radiograph.

Generate a detailed, grounded chest radiograph report for clinical review."""

        elif scan_type == "skin_cancer":
            scores_text = "\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
            )
            return f"""EXAM: Dermatoscopic image
AVAILABLE INPUT: One uploaded image only.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING CLASSIFIER OUTPUT (not a substitute for visual findings):
- Highest-scoring label: {result.top_label} ({result.confidence * 100:.1f}%)
- Class scores:
{scores_text}

DERMATOLOGY-SPECIFIC INSTRUCTIONS:
- Describe the visible lesion characteristics.
- Do not convert classifier confidence into staging or urgency.

Generate a detailed, grounded dermatoscopic report for clinical review."""

        elif getattr(result, "task_type", None) == "detection" or scan_type in {"brain_tumor", "bone_fracture"}:
            exam_name = "Brain MRI image" if scan_type == "brain_tumor" else "Bone X-ray image"
            bboxes = getattr(result, "bounding_boxes", [])
            if bboxes:
                det_text = "\n".join(f"  - {d['class']} (Confidence: {d['confidence']*100:.1f}%)" for d in bboxes)
            else:
                det_text = "  - 0 detections."
            return f"""EXAM: {exam_name}
AVAILABLE INPUT: One uploaded 2D image only.
CLINICAL HISTORY: Not provided.
COMPARISON: No prior study supplied.

SUPPORTING DETECTION OUTPUT (not a substitute for visual findings):
- Task Type: Detection
- Detections count: {len(bboxes)}
- Detections:
{det_text}

INSTRUCTIONS:
- Describe the visible anatomy and any abnormalities.
- Mention the detection count and classes if supported visually.
- Do not invent a clinical severity (e.g., Mild, Moderate, Severe).
- A bounding box is a localization aid, not a confirmed lesion.

Generate a detailed, grounded report for clinical review."""

        else:
            return f"""SCAN TYPE: Medical Image ({scan_type})
AI MODEL OUTPUT:
- Primary Finding: {result.top_label} (Confidence: {result.confidence * 100:.1f}%)
- Severity Assessment: {result.severity}

Generate a structured diagnostic report."""

    async def generate_report(
        self,
        result,
        scan_type: str,
        modality: str = "X-ray",
        patient_id: str = "P-",
        image=None,
    ) -> dict:
        if isinstance(result, dict):
            from types import SimpleNamespace
            result = SimpleNamespace(
                top_label=result.get("top_label") or "Diagnostic finding",
                confidence=float(result.get("confidence") or 0.0),
                all_scores=result.get("all_scores") or {},
                bounding_boxes=result.get("bounding_boxes") or result.get("detections") or [],
                task_type=result.get("task_type") or ("detection" if scan_type in {"brain_tumor", "bone_fracture"} else "classification"),
                severity=result.get("severity"),
                heatmap_target_label=result.get("heatmap_target_label"),
            )
        report_started = time.perf_counter()
        user_prompt = self._build_user_prompt(result, scan_type)
        llm_report = None
        llm_provider = "template"

        if self.gemini_service and image is not None:
            try:
                import io
                image_rgb = image.convert("RGB")
                buffer = io.BytesIO()
                image_rgb.save(buffer, format="JPEG", quality=90)
                image_bytes = buffer.getvalue()

                prompt = f"{SYSTEM_PROMPT}\n\nADDITIONAL SAFETY REQUIREMENTS:\n- This output is an unsigned preliminary draft for clinician verification.\n- If the image does not match the selected scan type, state that the image is not suitable for this workflow.\n- Use the uploaded image for clinical observations. Never expose the classifier, its confidence, model agreement, or concordance in the report prose.\n- For brain MRI, contrast status and pulse sequence are unknown. Do not call a lesion enhancing or make diffusion-, ADC-, FLAIR-, T1-, T2-, susceptibility-, or perfusion-specific claims.\n- Return only the required JSON object.\n\n{user_prompt}"

                report_obj = await self.gemini_service.generate_structured_report(prompt, image_bytes)
                if report_obj:
                    llm_report = report_obj.model_dump()
                    llm_provider = "gemini"
                    logger.info("Report generated successfully using Gemini.")
            except Exception as gemini_err:
                logger.warning("Gemini report generation failed: %s. Trying Groq fallback...", gemini_err)

        # Groq Fallback
        if llm_report is None and self.groq_service:
            try:
                groq_prompt = f"{SYSTEM_PROMPT}\n\nOutput strictly valid JSON with keys: technique, comparison, image_quality, findings, impression, differential_diagnosis, recommendations, critical_communication, patient_explanation.\n\n{user_prompt}"
                groq_data = await asyncio.to_thread(self.groq_service.generate_report, groq_prompt)
                if groq_data and isinstance(groq_data, dict):
                    llm_report = groq_data
                    llm_provider = "groq"
                    logger.info("Report generated successfully using Groq fallback.")
            except Exception as groq_err:
                logger.warning("Groq report generation failed: %s. Trying NVIDIA fallback...", groq_err)

        # NVIDIA Fallback
        if llm_report is None and self.nvidia_service:
            try:
                nv_prompt = f"{SYSTEM_PROMPT}\n\nOutput strictly valid JSON with keys: technique, comparison, image_quality, findings, impression, differential_diagnosis, recommendations, critical_communication, patient_explanation.\n\n{user_prompt}"
                nv_data = await asyncio.to_thread(self.nvidia_service.generate_report, nv_prompt)
                if nv_data and isinstance(nv_data, dict):
                    llm_report = nv_data
                    llm_provider = "nvidia"
                    logger.info("Report generated successfully using NVIDIA NIM fallback.")
            except Exception as nv_err:
                logger.warning("NVIDIA report generation failed: %s.", nv_err)

        nvidia_status = "unconfigured"
        if llm_report is not None and llm_provider == "gemini" and self.nvidia_service:
            import json
            diag_dict = {
                "top_label": result.top_label,
                "confidence": getattr(result, "confidence", 0.0),
                "all_scores": getattr(result, "all_scores", {}),
                "task_type": getattr(result, "task_type", "classification")
            }
            if getattr(result, "task_type", None) == "detection":
                diag_dict["bounding_boxes"] = getattr(result, "bounding_boxes", [])
            qa_res = self.nvidia_service.verify_report(diag_dict, json.dumps(llm_report))
            if qa_res is None:
                nvidia_status = "unavailable"
            else:
                nvidia_status = "checked"
                if not qa_res.passes or qa_res.recommendation.lower() == "flag":
                    logger.warning(f"NVIDIA QA flagged report: {qa_res.issues}")
                    llm_report = None

        if llm_report is None:
            llm_report = self._generate_template_report(result, scan_type)
            llm_provider = "template"
            nvidia_status = "bypassed"
        elif getattr(result, "task_type", "classification") == "classification" and not self._is_report_supported(
            llm_report,
            result,
            allow_visual_details=(llm_provider in {"gemini"}),
        ):
            logger.warning("LLM report contained unsupported findings. Falling back to template.")
            llm_report = self._generate_template_report(result, scan_type)
            llm_provider = "template"
            nvidia_status = "bypassed"

        llm_report = self._complete_report_sections(llm_report, scan_type)
        llm_report = self._ground_report_to_available_input(llm_report, scan_type)

        now = datetime.now()
        is_low_confidence = getattr(result, "is_low_confidence", False)
        heatmap_target_label = getattr(result, "heatmap_target_label", result.top_label)
        secondary_findings = getattr(result, "secondary_findings", [])

        if scan_type == "chest_xray":
            methodology = (
                "Classification was performed using the RAD-DINO ViT-B/14 chest-radiograph "
                "foundation encoder with a local three-class downstream head trained for Normal, "
                "Pneumonia, and Tuberculosis. The model processes 518x518 images and outputs a "
                "mutually-exclusive softmax score distribution. "
                "The accompanying heatmap uses the reference chest pipeline's lung-masked "
                "image saliency visualization in the same resize and center-crop coordinate frame. "
                "It is an explanatory image aid, not confirmed lesion segmentation. "
                f"Displayed classifier class: {heatmap_target_label}."
            )
        elif scan_type == "brain_mri":
            methodology = (
                "Classification was performed using an EfficientNetB3 convolutional neural network "
                "with ImageNet pretrained weights, fine-tuned via progressive 3-phase training "
                "for 4-class brain tumor classification (Glioma, Meningioma, No Tumor, Pituitary). "
                "Input images undergo brain contour cropping and are processed at 260x260 resolution. "
                "Test-Time Augmentation (TTA) is applied at inference for improved accuracy. "
                "Explainability was generated using class-logit Grad-CAM++ targeting "
                "the top_activation layer (9x9). "
                "The heatmap represents actual gradient-weighted activations from the trained model."
            )
        elif scan_type == "skin_cancer":
            methodology = "Classification was performed using a MobileNetV3-Large network for 7-class skin lesion classification. The accompanying heatmap uses Grad-CAM for localization."
        elif scan_type == "pneumonia":
            methodology = "Classification was performed using a MobileNetV3-Large network. The accompanying heatmap uses Grad-CAM for localization."
        elif scan_type == "brain_tumor":
            methodology = "Detection was performed using a YOLOv9m model. The accompanying image is a bounding-box overlay."
        elif scan_type == "bone_fracture":
            methodology = "Detection was performed using a YOLO11 model. The accompanying image is a bounding-box overlay."
        else:
            methodology = "AI model classification with model-attribution explainability."

        limitations = (
            "This AI system has inherent limitations: (1) The model was trained on a specific dataset "
            "and may not generalize to all patient populations or imaging equipment. "
            "(2) Multi-label classification confidence scores are not calibrated probabilities and "
            "should not be interpreted as disease prevalence. "
            "(3) Model-attribution heatmaps indicate model-influential regions but do not constitute "
            "radiologist-confirmed lesion localization. "
            "(4) The system cannot detect pathologies outside its training classes. "
            "(5) Image quality, positioning, and artifacts may affect model performance."
        )

        report_payload = {
            "patient_id": patient_id,
            "scan_date": now.strftime("%Y-%m-%d"),
            "scan_type": scan_type,
            "modality": modality,
            "top_label": result.top_label,
            "confidence": result.confidence,
            "clinical_history": "Not provided.",
            "technique": llm_report["technique"],
            "comparison": llm_report["comparison"],
            "image_quality": llm_report["image_quality"],
            "findings": llm_report.get("findings", "Findings not available."),
            "impression": llm_report.get("impression", "Impression not available."),
            "differential_diagnosis": llm_report["differential_diagnosis"],
            "recommendations": llm_report.get("recommendations", "Clinical correlation recommended."),
            "critical_communication": llm_report["critical_communication"],
            "patient_explanation": llm_report.get("patient_explanation", ""),
            "severity": result.severity,
            "all_scores": result.all_scores,
            "llm_provider": llm_provider,
            "disclaimer": DISCLAIMER,
            "generated_at": now.isoformat(),
            "is_low_confidence": is_low_confidence,
            "heatmap_target_label": heatmap_target_label,
            "secondary_findings": secondary_findings,
            "methodology": methodology,
            "limitations": limitations,
            "nvidia_qa_status": nvidia_status,
        }
        logger.info(
            "Clinical report generated via %s in %.0fms",
            llm_provider,
            (time.perf_counter() - report_started) * 1000,
        )
        return report_payload


    @classmethod






    def _generate_template_report(self, result, scan_type: str) -> dict:
        """Grounded fallback used when image-aware report generation is unavailable."""
        if isinstance(result, dict):
            from types import SimpleNamespace
            result = SimpleNamespace(
                top_label=result.get("top_label") or "Diagnostic finding",
                confidence=float(result.get("confidence") or 0.0),
                all_scores=result.get("all_scores") or {},
                bounding_boxes=result.get("bounding_boxes") or result.get("detections") or [],
                task_type=result.get("task_type") or ("detection" if scan_type in {"brain_tumor", "bone_fracture"} else "classification"),
                severity=result.get("severity"),
            )
        label = result.top_label
        score = float(result.confidence)
        secondary = [
            (name, float(value))
            for name, value in sorted((result.all_scores or {}).items(), key=lambda item: -item[1])
            if name != label and float(value) >= 0.20
        ][:3]
        differential = (
            "; ".join(f"{name} (secondary classifier score {value * 100:.1f}%)" for name, value in secondary)
            if secondary else "None based on the available automated signal."
        )

        if scan_type == "chest_xray":
            technique = (
                "Single chest radiograph submitted for review. Projection, positioning, exposure "
                "parameters, and additional views were not provided."
            )
            image_quality = (
                "Limited assessment: a formal image-quality and anatomy-by-anatomy visual review "
                "could not be generated. Subtle findings may not be represented."
            )
            if label == "Normal":
                findings = (
                    "LUNGS/AIRWAYS: No trained thoracic abnormality crossed the configured reporting threshold.\n"
                    "PLEURA: Not independently assessed in this fallback report.\n"
                    "CARDIOMEDIASTINAL SILHOUETTE: Not independently assessed in this fallback report.\n"
                    "HILA: Not independently assessed in this fallback report.\n"
                    "BONES/SOFT TISSUES: Not independently assessed in this fallback report.\n"
                    "SUPPORT DEVICES: Not independently assessed in this fallback report."
                )
                impression = (
                    "1. No model-supported thoracic abnormality above the reporting threshold.\n"
                    "2. This limited result does not establish a normal chest radiograph."
                )
            else:
                findings = (
                    f"LUNGS/AIRWAYS: Automated analysis produced its strongest signal for {label} "
                    f"({score * 100:.1f}%); independent visual confirmation is unavailable in this fallback report.\n"
                    "PLEURA: Not independently assessed except where represented by the classifier label above.\n"
                    "CARDIOMEDIASTINAL SILHOUETTE: Not independently assessed except where represented by the classifier label above.\n"
                    "HILA: Not independently assessed.\n"
                    "BONES/SOFT TISSUES: Not independently assessed.\n"
                    "SUPPORT DEVICES: Not independently assessed."
                )
                impression = (
                    f"1. Indeterminate chest radiograph with an automated signal for {label}; "
                    "confirmation on direct image review is required."
                )
            recommendations = (
                "Direct review of the original radiograph with clinical correlation is recommended. "
                "Obtain additional or follow-up imaging only when indicated by symptoms, examination, and the verified imaging finding."
            )
        elif scan_type == "brain_mri":
            technique = (
                "Single brain MRI image submitted for review. Sequence, plane, contrast status, and "
                "the remainder of the diagnostic MRI series were not provided."
            )
            image_quality = (
                "Severely limited examination because only one exported image is available. A complete "
                "assessment of the brain, diffusion, enhancement, hemorrhage, and small lesions is not possible."
            )
            if label == "No Tumor":
                findings = (
                    "BRAIN PARENCHYMA: No lesion in the three trained tumor categories was supported above the classification threshold.\n"
                    "EXTRA-AXIAL SPACES: Not completely assessable on the supplied image.\n"
                    "VENTRICLES/MASS EFFECT: Not completely assessable on the supplied image.\n"
                    "SELLA: Not completely assessable on the supplied image.\n"
                    "VISIBLE POSTERIOR FOSSA: Not completely assessable on the supplied image."
                )
                impression = (
                    "1. No model-supported glioma, meningioma, or pituitary tumor on this limited single-image analysis.\n"
                    "2. Other intracranial pathology and lesions not represented in the trained categories are not excluded."
                )
            else:
                findings = (
                    f"BRAIN PARENCHYMA: Automated analysis produced its strongest category signal for {label} "
                    f"({score * 100:.1f}%); independent lesion characterization is unavailable in this fallback report.\n"
                    "EXTRA-AXIAL SPACES: Not completely assessable on the supplied image.\n"
                    "VENTRICLES/MASS EFFECT: Not completely assessable on the supplied image.\n"
                    "SELLA: Not completely assessable beyond the category signal above.\n"
                    "VISIBLE POSTERIOR FOSSA: Not completely assessable on the supplied image."
                )
                impression = (
                    f"1. Indeterminate abnormality with an automated category signal for {label}; "
                    "the single supplied image is insufficient for a definitive tumor diagnosis or characterization."
                )
            recommendations = (
                "Review the complete diagnostic MRI examination, including all available sequences and prior studies. "
                "Further imaging or specialty referral should be based on verified imaging findings and the clinical presentation."
            )
        elif scan_type in {"lung_ct", "kidney_us"}:
            exam = "lung CT" if scan_type == "lung_ct" else "kidney ultrasound"
            technique = (
                f"Single exported {exam} image submitted for limited review. The complete examination, "
                "acquisition parameters, and additional views are not available."
            )
            image_quality = (
                "Diagnostic completeness cannot be established from one exported image; subtle or out-of-frame findings may not be represented."
            )
            findings = (
                f"Automated analysis produced its strongest category signal for {label} ({score * 100:.1f}%). "
                "Independent characterization is unavailable in this template fallback."
            )
            impression = (
                f"1. Indeterminate {exam} category signal for {label}; direct review of the complete source examination is required."
            )
            recommendations = (
                f"Review the complete {exam} study with clinical correlation. Management should be based on verified imaging findings."
            )
        elif scan_type in {"pneumonia", "skin_cancer"}:
            exam = "chest radiograph" if scan_type == "pneumonia" else "dermatoscopic image"
            technique = f"Single exported {exam} submitted for limited review."
            image_quality = "Diagnostic completeness cannot be established from one exported image."
            findings = f"Automated analysis produced its strongest category signal for {label} ({score * 100:.1f}%)."
            impression = f"1. Indeterminate {exam} category signal for {label}."
            recommendations = f"Review the complete study with clinical correlation."
        elif scan_type in {"brain_tumor", "bone_fracture"} or getattr(result, "task_type", None) == "detection":
            exam = "brain MRI" if scan_type == "brain_tumor" else "bone X-ray"
            technique = f"Single exported {exam} submitted for limited review."
            image_quality = "Diagnostic completeness cannot be established from one exported image."
            bboxes = getattr(result, "bounding_boxes", [])
            findings = f"Automated analysis detected {len(bboxes)} bounding boxes. Primary detection: {label}."
            impression = f"1. {len(bboxes)} automated detections."
            recommendations = f"Review the complete study with clinical correlation."
        else:
            technique = "Single medical image submitted for limited review."
            image_quality = "Completeness and diagnostic adequacy cannot be established from the supplied input."
            findings = f"Automated analysis produced its strongest signal for {label} ({score * 100:.1f}%)."
            impression = f"1. Indeterminate automated finding: {label}."
            recommendations = "Direct clinician review and clinical correlation are recommended."

        return {
            "technique": technique,
            "comparison": "No prior imaging was supplied for comparison.",
            "image_quality": image_quality,
            "findings": findings,
            "impression": impression,
            "differential_diagnosis": differential,
            "recommendations": recommendations,
            "critical_communication": "Not applicable — no critical or emergent finding requiring direct communication.",
            "patient_explanation": "",
        }

    def _generate_legacy_template_report(self, result, scan_type: str) -> dict:
        """
        Template-based report generation (fallback when no LLM API key).
        Produces clinically structured text without LLM enhancement.
        """
        label = result.top_label
        conf = result.confidence
        severity = result.severity
        is_low_confidence = getattr(result, "is_low_confidence", False)
        secondary_findings = getattr(result, "secondary_findings", [])
        heatmap_target_label = getattr(result, "heatmap_target_label", label)

        if scan_type == "chest_xray":
            if label == "Normal":
                findings = (
                    "TECHNIQUE: Frontal chest radiograph was analyzed using the MedRittAI "
                    "RAD-DINO foundation encoder with a three-class chest classification head. "
                    "The reference lung-masked image saliency heatmap was computed "
                    f"for the selected output ({heatmap_target_label}).\n\n"
                    "FINDINGS: The AI classifier did not identify any model-supported acute "
                    "cardiopulmonary abnormality above the configured reporting threshold. "
                    "The Normal class received the highest softmax score. The attribution map shows where the "
                    "model focused its analysis, but no region triggered a pathology classification. "
                    "This automated result does not exclude subtle or early-stage pathology that "
                    "falls below model sensitivity."
                )
                impression = (
                    "No model-supported acute chest X-ray abnormality detected. "
                    "The AI attribution map did not identify a dominant model-influential region. "
                    "Clinical correlation with patient history and symptoms is recommended."
                )
                recommendations = (
                    "1. Radiologist review is recommended as clinically indicated.\n"
                    "2. Correlate with clinical symptoms, patient history, and physical examination.\n"
                    "3. Consider repeat imaging if clinical suspicion remains high despite negative AI result.\n"
                    "4. This AI system cannot detect all pathologies; absence of a finding does not exclude disease."
                )
            else:
                # Build secondary findings text
                secondary_text = ""
                if secondary_findings:
                    sec_items = [f"{sf['label']} ({sf['score'] * 100:.0f}%)" for sf in secondary_findings[:3]]
                    secondary_text = (
                        f"\n\nDIFFERENTIAL CONSIDERATIONS: The model also produced elevated "
                        f"scores for: {', '.join(sec_items)}. These may represent co-existing "
                        f"pathology or classification overlap and should be considered during "
                        f"radiologist review."
                    )

                confidence_note = ""
                if is_low_confidence:
                    confidence_note = (
                        " IMPORTANT: The model score is below the preferred decision margin, "
                        "indicating significant uncertainty. The finding should be interpreted "
                        "with caution and requires careful radiologist correlation. "
                        "Low-confidence AI results have higher rates of false positives."
                    )

                findings = (
                    "TECHNIQUE: Frontal chest radiograph was analyzed using the MedRittAI "
                    "RAD-DINO foundation encoder with a three-class chest classification head. "
                    "The reference lung-masked image saliency heatmap was computed "
                    f"targeting the {heatmap_target_label} class to visualize the region of "
                    "model attention most relevant to the primary finding.\n\n"
                    f"FINDINGS: The AI chest X-ray model produced findings suggestive of "
                    f"{label} with an uncalibrated softmax score of {conf * 100:.1f}%. "
                    f"The chest heatmap highlights visually salient regions inside the lung mask. "
                    f"The highlighted region should be correlated with clinical findings and "
                    f"is NOT equivalent to radiologist-confirmed lesion localization. "
                    f"The findings are assessed as {severity.lower()} based on model "
                    f"confidence scoring.{confidence_note}{secondary_text}"
                )
                impression = (
                    f"AI findings suggestive of {label} "
                    f"(confidence: {conf * 100:.1f}%, severity: {severity}). "
                    + ("Low model confidence — interpret with caution. " if is_low_confidence else "")
                    + "Clinical correlation with patient history, symptoms, and physical "
                    "examination is essential. Radiologist review required before clinical action."
                )
                recommendations = (
                    f"1. Formal radiologist review and interpretation is required.\n"
                    f"2. Correlate with clinical history, symptoms, and physical examination findings.\n"
                    f"3. {'Urgent clinical attention may be warranted given severity assessment. ' if severity == 'Severe' else ''}"
                    f"Consider follow-up imaging as clinically indicated.\n"
                    f"4. If findings are discordant with clinical presentation, consider "
                    f"alternative diagnoses or additional imaging modalities.\n"
                    f"5. AI classification should not be used as the sole basis for clinical decisions."
                )

        elif scan_type == "brain_mri":
            technique = (
                "TECHNIQUE: Brain MRI was analyzed using the MedRittAI EfficientNetB3 "
                "4-class tumor classifier (Glioma, Meningioma, No Tumor, Pituitary) with "
                "multi-scale Grad-CAM++ explainability. Input preprocessing includes brain "
                "contour cropping and EfficientNet-specific normalization at 260×260 resolution. "
                "Test-Time Augmentation was applied for improved accuracy."
            )

            # Build differential text from all scores
            sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])
            diff_items = [f"{s[0]} ({s[1] * 100:.0f}%)" for s in sorted_scores if s[1] >= 0.05]
            diff_text = ", ".join(diff_items) if diff_items else "No significant scores"

            if label == "No Tumor":
                findings = (
                    f"{technique}\n\n"
                    "FINDINGS: Brain MRI was reviewed. The AI classifier did not identify any "
                    "model-supported intracranial mass lesion. All tumor class probabilities "
                    "(Glioma, Meningioma, Pituitary) remained below the classification threshold. "
                    f"Class distribution: {diff_text}. "
                    "The Grad-CAM++ attention map did not localize a significant region of "
                    "abnormality. This automated result does not exclude all intracranial "
                    "pathology, particularly small, diffuse, or non-enhancing lesions outside "
                    "the model's training distribution."
                )
                impression = (
                    "No evidence of intracranial tumor on AI-assisted MRI analysis "
                    "(4-class EfficientNetB3 classifier). Clinical correlation recommended."
                )
                recommendations = (
                    "1. Routine clinical follow-up as indicated.\n"
                    "2. Radiologist review recommended if clinical symptoms persist.\n"
                    "3. AI classification has limited sensitivity for small, diffuse, or "
                    "non-enhancing lesions.\n"
                    "4. Consider contrast-enhanced MRI if clinical suspicion remains high."
                )
            elif label == "Glioma":
                findings = (
                    f"{technique}\n\n"
                    f"FINDINGS: Brain MRI analysis reveals findings suggestive of a glioma "
                    f"(AI confidence: {conf * 100:.1f}%). Gliomas are primary brain tumors "
                    f"arising from glial cells and may range from low-grade (WHO Grade I-II) "
                    f"to high-grade (WHO Grade III-IV, glioblastoma). "
                    f"The Grad-CAM++ attention map highlights the region contributing most "
                    f"to this classification. The AI model does not determine tumor grade, "
                    f"molecular subtype, or extent of infiltration. "
                    f"Differential scores: {diff_text}. "
                    f"The findings are assessed as {severity.lower()} based on model "
                    f"confidence scoring."
                )
                impression = (
                    f"AI findings suggestive of glioma "
                    f"(confidence: {conf * 100:.1f}%, severity: {severity}). "
                    f"URGENT neurosurgical and neuroradiology consultation recommended. "
                    f"Tumor grading and molecular characterization require histopathological analysis."
                )
                recommendations = (
                    "1. URGENT: Neurosurgical consultation recommended.\n"
                    "2. Formal neuroradiology review with contrast-enhanced MRI (T1 post-Gd, "
                    "FLAIR, DWI, perfusion) for characterization and grading.\n"
                    "3. Correlation with clinical symptoms and neurological examination is essential.\n"
                    "4. Consider advanced imaging (MR spectroscopy, perfusion) for grading.\n"
                    "5. Stereotactic biopsy or surgical planning if lesion is confirmed."
                )
            elif label == "Meningioma":
                findings = (
                    f"{technique}\n\n"
                    f"FINDINGS: Brain MRI analysis reveals findings suggestive of a meningioma "
                    f"(AI confidence: {conf * 100:.1f}%). Meningiomas are typically benign "
                    f"(WHO Grade I) extra-axial tumors arising from the meninges, often "
                    f"well-circumscribed with dural attachment. "
                    f"The Grad-CAM++ attention map highlights the region contributing most "
                    f"to this classification. The AI model does not determine tumor grade, "
                    f"size, or dural involvement. "
                    f"Differential scores: {diff_text}. "
                    f"The findings are assessed as {severity.lower()} based on model "
                    f"confidence scoring."
                )
                impression = (
                    f"AI findings suggestive of meningioma "
                    f"(confidence: {conf * 100:.1f}%, severity: {severity}). "
                    f"Neuroradiology review recommended for confirmation and characterization."
                )
                recommendations = (
                    "1. Neuroradiology review with contrast-enhanced MRI for confirmation.\n"
                    "2. Assess for dural tail sign, calcification, and extra-axial characteristics.\n"
                    "3. Neurosurgical consultation for management planning.\n"
                    "4. Consider observation with serial imaging for small, asymptomatic lesions.\n"
                    "5. Correlate with clinical symptoms and neurological examination."
                )
            elif label == "Pituitary":
                findings = (
                    f"{technique}\n\n"
                    f"FINDINGS: Brain MRI analysis reveals findings suggestive of a pituitary "
                    f"tumor (AI confidence: {conf * 100:.1f}%). Pituitary tumors are typically "
                    f"adenomas located in the sellar/suprasellar region, and may be functioning "
                    f"(hormone-secreting) or non-functioning. "
                    f"The Grad-CAM++ attention map highlights the region contributing most "
                    f"to this classification. The AI model does not determine tumor size, "
                    f"hormonal activity, or suprasellar extension. "
                    f"Differential scores: {diff_text}. "
                    f"The findings are assessed as {severity.lower()} based on model "
                    f"confidence scoring."
                )
                impression = (
                    f"AI findings suggestive of pituitary tumor "
                    f"(confidence: {conf * 100:.1f}%, severity: {severity}). "
                    f"Endocrinology and neuroradiology consultation recommended."
                )
                recommendations = (
                    "1. Endocrinology consultation for hormonal evaluation (prolactin, GH, "
                    "ACTH, TSH, gonadotropins).\n"
                    "2. Dedicated pituitary MRI with thin-section coronal T1 pre/post-contrast.\n"
                    "3. Visual field testing (perimetry) to assess for chiasmal compression.\n"
                    "4. Neurosurgical consultation if mass effect or hormonal excess confirmed.\n"
                    "5. Correlate with clinical symptoms (headache, visual changes, hormonal symptoms)."
                )
            else:
                # Fallback for any unexpected label
                findings = (
                    f"{technique}\n\n"
                    f"FINDINGS: Brain MRI analysis reveals findings classified as {label} "
                    f"(AI confidence: {conf * 100:.1f}%). "
                    f"Differential scores: {diff_text}. "
                    f"The findings are assessed as {severity.lower()} based on model "
                    f"confidence scoring."
                )
                impression = (
                    f"AI findings suggest {label} "
                    f"(confidence: {conf * 100:.1f}%, severity: {severity}). "
                    f"Neuroradiology review recommended."
                )
                recommendations = (
                    "1. Neuroradiology review recommended.\n"
                    "2. Correlate with clinical symptoms and neurological examination.\n"
                    "3. Consider contrast-enhanced MRI for further characterization."
                )
        else:
            findings = f"AI analysis identifies {label} with {conf * 100:.1f}% confidence."
            impression = f"Findings suggestive of {label}. Severity: {severity}."
            recommendations = "Clinical correlation recommended."

        return {
            "findings": findings,
            "impression": impression,
            "recommendations": recommendations,
        }

    @staticmethod
    def _parse_json_report(content: str) -> dict:
        """Parse provider JSON while tolerating accidental code fences."""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Report response must be a JSON object")
        fields = (
            "technique", "comparison", "image_quality", "findings", "impression",
            "differential_diagnosis", "recommendations", "critical_communication",
            "patient_explanation",
        )
        report = {field: str(parsed.get(field, "")).strip() for field in fields}
        if not report["findings"] or not report["impression"]:
            raise ValueError("Report response is missing required clinical sections")
        return report

    @staticmethod
    def _complete_report_sections(report: dict, scan_type: str) -> dict:
        """Normalize provider and legacy template output into the clinical schema."""
        technique_defaults = {
            "brain_mri": "Single brain MRI image submitted; sequence, plane, contrast status, and complete series are not provided.",
            "chest_xray": "Single chest radiograph submitted; projection, positioning, and additional views are not provided.",
            "lung_ct": "Single lung CT image submitted; acquisition details and complete series are not provided.",
            "kidney_us": "Single kidney ultrasound image submitted; acquisition details and complete study are not provided.",
            "pneumonia": "Single chest radiograph submitted.",
            "skin_cancer": "Single dermatoscopic image submitted.",
            "brain_tumor": "Single brain MRI image submitted.",
            "bone_fracture": "Single bone X-ray image submitted.",

        }
        default_technique = technique_defaults.get(scan_type, "Single medical image submitted for limited review.")
        defaults = {
            "technique": default_technique,
            "comparison": "No prior imaging was supplied for comparison.",
            "image_quality": "Diagnostic adequacy is limited by review of a single supplied image.",
            "findings": "Findings are not available.",
            "impression": "No impression was generated.",
            "differential_diagnosis": "None based on the supplied image.",
            "recommendations": "Clinical correlation and direct image review are recommended.",
            "critical_communication": "Not applicable — no critical or emergent finding requiring direct communication.",
            "patient_explanation": "",
        }
        return {
            key: str(report.get(key) or fallback).strip()
            for key, fallback in defaults.items()
        }

    @staticmethod
    def _sanitize_generated_clinical_text(value: str, scan_type: str) -> str:
        """Remove machine-facing and unsupported acquisition claims from generated prose."""
        text = str(value or "").strip()
        if not text:
            return text

        def drop_sentences(source: str, pattern: re.Pattern) -> str:
            cleaned_lines = []
            for line in source.splitlines() or [source]:
                kept = [
                    sentence
                    for sentence in re.split(r"(?<=[.!?])\s+", line.strip())
                    if sentence and not pattern.search(sentence)
                ]
                if kept:
                    cleaned_lines.append(" ".join(kept))
            return "\n".join(cleaned_lines)

        # A provider occasionally appends model agreement to an otherwise useful
        # clinical sentence. Remove the clause while preserving the observation.
        text = re.sub(
            r",?\s*(?:demonstrating|showing|with)?\s*concordance\s+with\s+(?:the\s+)?"
            r"(?:supportive\s+)?(?:classifier|model)(?:\s+(?:finding|output|result))?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r",?\s*(?:as\s+)?(?:supported|identified|predicted)\s+by\s+(?:the\s+)?"
            r"(?:classifier|model|automated analysis)",
            "",
            text,
            flags=re.IGNORECASE,
        )

        if scan_type == "brain_mri":
            # There is no DICOM series metadata at this stage. Keep the lesion
            # description, but strip acquisition-dependent adjectives and claims.
            text = re.sub(
                r"\b(?:avidly|mildly|moderately|strongly|homogeneously|heterogeneously)?\s*enhancing\b",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bno\s+(?:abnormal\s+)?enhancement\b|\black\s+of\s+enhancement\b",
                "contrast behavior cannot be assessed",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b(?:post[- ]?contrast|gadolinium(?:-enhanced)?|contrast uptake|enhancement)\b",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b(?:axial|coronal|sagittal)\s+(?:image|slice|view)\b",
                "supplied image",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bmeasur(?:es|ing)\s+(?:approximately\s+)?\d+(?:\.\d+)?\s*(?:mm|cm)\b",
                "cannot be reliably measured on the supplied image",
                text,
                flags=re.IGNORECASE,
            )

            # Drop whole sentences that still assert a sequence-dependent
            # observation. A recommendation may name a future sequence, so this
            # helper is applied only to generated findings/impression/differential.
            unsupported_sequence = re.compile(
                r"\b(?:restricted diffusion|diffusion restriction|ADC|DWI|FLAIR|SWI|"
                r"T1[- ]weighted|T2[- ]weighted|susceptibility|perfusion)\b",
                re.IGNORECASE,
            )
            text = drop_sentences(text, unsupported_sequence)

        # Clinical prose must read like a report, not an explanation of the
        # software. If a residual machine-facing sentence remains, omit it.
        machine_facing = re.compile(
            r"\b(?:classifier|model[- ]supported|model confidence|model agreement|"
            r"automated (?:analysis|signal|result|classification|finding)|AI|"
            r"artificial intelligence|Grad-CAM|model attribution|token attribution|attention map|heatmap)\b",
            re.IGNORECASE,
        )
        text = drop_sentences(text, machine_facing)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(
            r",\s+(?=(?:mass|lesion|abnormality|collection|opacity)\b)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        anatomy_labels = (
            "LUNGS/AIRWAYS|PLEURA|CARDIOMEDIASTINAL SILHOUETTE|HILA|"
            "BONES/SOFT TISSUES|SUPPORT DEVICES|BRAIN PARENCHYMA|"
            "EXTRA-AXIAL SPACES|VENTRICLES/MASS EFFECT|SELLA|"
            "VISIBLE POSTERIOR FOSSA"
        )
        text = re.sub(
            rf"(?<!^)\s+(?=(?:{anatomy_labels}):)",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        return text.strip(" ,;")

    @classmethod
    def _ground_report_to_available_input(cls, report: dict, scan_type: str) -> dict:
        """Apply deterministic study metadata and a final clinical grounding gate."""
        grounded = dict(report)
        grounded["comparison"] = "No prior imaging was supplied for comparison."

        if scan_type == "brain_mri":
            grounded["technique"] = (
                "Single uploaded 2D brain MRI image. Imaging plane, pulse sequence, "
                "contrast status, and the complete multiplanar series are not available."
            )
            grounded["image_quality"] = (
                "Limited diagnostic assessment because only one exported image is available. "
                "Full brain coverage, sequence-dependent tissue characteristics, contrast "
                "behavior, hemorrhage, and small lesions cannot be assessed reliably."
            )
        elif scan_type == "chest_xray":
            grounded["technique"] = (
                "Single uploaded frontal chest radiograph. Projection, positioning, exposure "
                "parameters, and additional views are not available."
            )
            grounded["image_quality"] = (
                "Interpretation is limited to one exported radiograph; projection, positioning, "
                "exposure parameters, and the complete examination cannot be independently verified."
            )
        elif scan_type in {"lung_ct", "kidney_us"}:
            exam_name = "lung CT" if scan_type == "lung_ct" else "kidney ultrasound"
            grounded["technique"] = (
                f"Single uploaded 2D {exam_name} image. Acquisition details, additional views, "
                "and the complete source examination are not available."
            )
            grounded["image_quality"] = (
                "Limited diagnostic assessment because only one exported image is available; "
                "subtle and out-of-frame findings cannot be excluded."
            )
        elif scan_type in {"pneumonia", "skin_cancer", "brain_tumor", "bone_fracture"}:
            grounded["technique"] = f"Single uploaded {scan_type.replace('_', ' ')} image."
            grounded["image_quality"] = "Limited diagnostic assessment."


        if not str(grounded.get("recommendations") or "").strip():
            grounded["recommendations"] = (
                "Review the complete source examination and correlate with the clinical presentation."
            )
        if not str(grounded.get("critical_communication") or "").strip():
            grounded["critical_communication"] = (
                "Not applicable — no critical or emergent finding requiring direct communication."
            )

        for field in ("findings", "impression", "differential_diagnosis"):
            cleaned = cls._sanitize_generated_clinical_text(
                grounded.get(field, ""), scan_type
            )
            safe_fallbacks = {
                "findings": (
                    "A complete visual interpretation cannot be provided from the available "
                    "single image. Direct review of the source examination is required."
                ),
                "impression": (
                    "1. Limited single-image assessment; definitive interpretation requires "
                    "review of the complete source examination."
                ),
                "differential_diagnosis": (
                    "No reliable differential can be established from the supplied image alone."
                ),
            }
            grounded[field] = cleaned or safe_fallbacks[field]

        return grounded

    @staticmethod
    def _is_report_supported(report: dict, result, allow_visual_details: bool = False) -> bool:
        """Reject text-only reports that invent findings absent from their supplied context."""
        text = " ".join(
            str(report.get(key, ""))
            for key in (
                "findings", "impression", "differential_diagnosis",
                "recommendations", "critical_communication",
            )
        ).lower()

        # MAIRA-2 and Gemini inspect the radiograph itself. Their visual
        # observations are allowed to differ from the supporting classifier;
        # forcing agreement here discarded precisely the image evidence these
        # providers were added to supply. Deterministic metadata grounding and
        # clinical-text sanitization are still applied after this gate.
        if allow_visual_details:
            return True

        # If there are no all_scores, we can't reliably validate the text against classes
        if not getattr(result, "all_scores", None):
            return True

        supported = {str(result.top_label).lower(), "normal", "no finding"}
        supported.update(
            label.lower()
            for label, score in result.all_scores.items()
            if score >= 0.20
        )

        for label in result.all_scores.keys():
            normalized = label.lower()
            if normalized in text and normalized not in supported:
                return False

        blocked_terms = [
            "right upper lobe", "right middle lobe", "right lower lobe",
            "left upper lobe", "left lower lobe", "apical", "basilar",
            " cm", " mm", "tube", "line placement", "prior study",
        ]
        return not any(term in text for term in blocked_terms)

    # ============================================================
    # PATIENT-FRIENDLY SUMMARY
    # ============================================================

    PATIENT_SUMMARY_PROMPT = """Convert the clinical imaging draft below into a safe English explanation for the patient.

RULES:
- Use calm, direct, everyday English and short sentences.
- Preserve uncertainty. A suspected finding must remain suspected; never turn it into a confirmed diagnosis.
- Clearly separate: what the image may show, what that could mean, limitations, and the next step.
- Do not add symptoms, prognosis, treatment, urgency, or reassurance that is absent from the report.
- Include urgent-care advice only when the report's critical communication or recommendations support it.
- Do not mention model/provider names, prompts, confidence scores, heatmaps, or technical implementation.
- Explain essential medical terms once in plain language instead of deleting information the patient needs.
- Do not say the scan is normal when the report only says that no trained abnormality was detected.
- Use these four headings exactly: What the scan may show; What this means; What happens next; Important limitation.
- Write 4 short paragraphs, one under each heading. Do not use markdown bullets or JSON.
"""

    LANGUAGE_CODES = {
        "English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN",
        "Telugu": "te-IN", "Marathi": "mr-IN", "Bengali": "bn-IN",
        "Kannada": "kn-IN", "Gujarati": "gu-IN", "Malayalam": "ml-IN",
        "Punjabi": "pa-IN", "Urdu": "ur-IN",
    }

    async def generate_patient_report(
        self,
        report_data: dict,
        language: str = "English",
    ) -> str:
        """
        Generate a patient-friendly summary of the medical report.

        Creates a grounded English explanation first, then translates that fixed
        text. Translation is deliberately separated from clinical generation.

        Args:
            report_data: The full clinical report data dict
            language: Target language (e.g., "Hindi", "Tamil", "English")

        Returns:
            Plain text patient-friendly summary string
        """
        requested_language = language if language in self.LANGUAGE_CODES else "English"
        english = await self._generate_patient_explanation_english(report_data)
        if requested_language == "English":
            return english

        translated = None
        if self.sarvam_key and not self.sarvam_translation_disabled:
            translated = await self._translate_with_sarvam(
                english, self.LANGUAGE_CODES[requested_language]
            )
        if translated is None and self.gemini_key:
            translated = await self._translate_with_gemini(english, requested_language)
        if translated:
            return translated

        return f"[{requested_language} translation is temporarily unavailable.]\n\n{english}"

    async def _generate_patient_explanation_english(self, report_data: dict) -> str:
        stored = str(report_data.get("patient_explanation") or "").strip()
        if stored:
            return stored

        scan_type = report_data.get("scan_type", "medical scan")
        scan_label = {
            "brain_mri": "brain MRI image",
            "chest_xray": "chest X-ray",
            "lung_ct": "lung CT image",
            "kidney_us": "kidney ultrasound image",
            "pneumonia": "chest X-ray",
            "skin_cancer": "dermatoscopy image",
            "brain_tumor": "brain MRI image",
            "bone_fracture": "bone X-ray",

        }.get(scan_type, "medical image")
        return self._patient_summary_template(report_data, scan_label)

    async def _translate_with_sarvam(self, text: str, target_code: str) -> Optional[str]:
        """Translate fixed English patient text; scan data is never sent here."""
        try:
            import httpx

            chunks = self._chunk_translation_text(text)
            translated_chunks = []
            async with httpx.AsyncClient(timeout=12.0) as client:
                for chunk in chunks:
                    response = await client.post(
                        "https://api.sarvam.ai/translate",
                        headers={
                            "api-subscription-key": self.sarvam_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "input": chunk,
                            "source_language_code": "en-IN",
                            "target_language_code": target_code,
                            "model": self.sarvam_translate_model,
                            "mode": "formal",
                        },
                    )
                    response.raise_for_status()
                    translated = str(response.json().get("translated_text", "")).strip()
                    if not translated:
                        raise ValueError("Translation response contained no text")
                    translated_chunks.append(translated)
            logger.info("Patient explanation translated to %s", target_code)
            return "\n\n".join(translated_chunks)
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 402, 403}:
                self.sarvam_translation_disabled = True
                logger.warning(
                    "Primary translation disabled until backend restart after HTTP %s",
                    status_code,
                )
            logger.warning("Primary patient translation failed: %s", exc)
            return None

    async def _translate_with_gemini(self, text: str, language: str) -> Optional[str]:
        prompt = f"""Translate the patient explanation below from English into {language}.
Preserve all four headings, meaning, uncertainty, paragraph order, and safety language exactly.
Do not add, remove, summarize, diagnose, or explain anything. Output only the translation.

{text}"""
        if self.gemini_service:
            return await self.gemini_service.generate_text(prompt)
        return None


    @staticmethod
    def _chunk_translation_text(text: str, limit: int = 1900) -> list[str]:
        """Split long text at paragraph/sentence boundaries for the 2,000-char API limit."""
        import re

        if len(text) <= limit:
            return [text]
        units = re.split(r"(?<=\.)\s+|\n\s*\n", text.strip())
        chunks: list[str] = []
        current = ""
        for unit in (part.strip() for part in units if part.strip()):
            if len(unit) > limit:
                pieces = [unit[index:index + limit] for index in range(0, len(unit), limit)]
            else:
                pieces = [unit]
            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if len(candidate) > limit and current:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _patient_summary_template(report_data: dict, scan_type: str) -> str:
        """Immediate plain-English explanation that preserves diagnostic uncertainty."""
        label = str(report_data.get("top_label") or "Unknown").strip()
        explanations = {
            "Normal": "The Normal class scored highest among the three patterns checked; this does not exclude other conditions.",
            "Atelectasis": "Part of a lung may not be expanding as fully as expected.",
            "Cardiomegaly": "The outline of the heart may look larger than expected on this image.",
            "Effusion": "There may be fluid in the space around a lung.",
            "Pleural Effusion": "There may be fluid in the space around a lung.",
            "Infiltration": "An area of the lung may look denser than expected, which has several possible causes.",
            "Lung Opacity": "An area of the lung may look denser than expected, which has several possible causes.",
            "Mass": "The image may contain a larger focal area that needs direct medical review.",
            "Lung Lesion": "The image may contain a focal area that needs direct medical review.",
            "Nodule": "The image may contain a small rounded spot that needs direct medical review.",
            "Pneumonia": "The image may show a lung pattern that can occur with infection.",
            "Tuberculosis": "The image pattern may be compatible with tuberculosis, but clinical and microbiological confirmation are required.",
            "Pneumothorax": "There may be air around a lung, which can prevent the lung from fully expanding.",
            "Consolidation": "Part of a lung may be filled with fluid or inflammatory material rather than air.",
            "Edema": "There may be extra fluid within the lungs.",
            "Emphysema": "The lungs may show changes associated with long-term damage to their air spaces.",
            "Fibrosis": "The lungs may show areas of scarring.",
            "Fracture": "A visible bone may show a possible break.",
            "No Tumor": "No pattern from the three brain-tumor groups checked by this system was strongly identified.",
            "Glioma": "The image may show a growth pattern arising from brain tissue, sometimes called a glioma.",
            "Meningioma": "The image may show a growth pattern arising from the covering around the brain, sometimes called a meningioma.",
            "Pituitary": "The image may show a growth near the small hormone-producing gland beneath the brain.",
        }
        finding = explanations.get(
            label,
            "The image contains a possible finding that needs direct review by your treating doctor.",
        )
        critical = str(report_data.get("critical_communication") or "").lower()
        if critical and "no critical" not in critical:
            next_step = (
                "The report marks this for prompt communication. Contact your treating team now so they can "
                "review the scan and tell you what action is appropriate."
            )
        else:
            next_step = (
                "Please discuss the complete scan with your treating doctor. They will compare this result with "
                "your symptoms, examination, and any earlier scans before deciding whether another test is needed."
            )
        limitation = (
            "Only one brain MRI image was available here, not the complete set of MRI images. Important details "
            "may therefore be missing."
            if report_data.get("scan_type") == "brain_mri"
            else "This explanation is based on the supplied chest image and the limited conditions the system checks. "
            "Subtle or unrelated problems may not be represented."
        )
        return (
            f"What the scan may show\n{finding}\n\n"
            f"What this means\nThis is a possible imaging finding, not a confirmed diagnosis. The label “{label}” "
            "does not by itself determine how serious the condition is or what treatment you need.\n\n"
            f"What happens next\n{next_step}\n\n"
            f"Important limitation\n{limitation}"
        )

    async def generate_patient_summary(
        self,
        report_data: dict,
        language: str = "English",
    ) -> str:
        """Backward-compatible alias for older callers."""
        return await self.generate_patient_report(report_data, language)
