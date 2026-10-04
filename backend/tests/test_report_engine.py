import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from services.llm_report_engine import LLMReportEngine


def result(label="Glioma", confidence=0.91, severity="Severe"):
    return SimpleNamespace(
        top_label=label,
        confidence=confidence,
        severity=severity,
        all_scores={"Glioma": confidence, "Meningioma": 0.06, "No Tumor": 0.02, "Pituitary": 0.01},
        is_low_confidence=confidence < 0.5,
        heatmap_target_label=label,
        secondary_findings=[],
    )


class ReportSchemaTests(unittest.TestCase):
    def test_image_aware_chest_report_may_differ_from_classifier(self):
        classification = result(label="Pneumonia", confidence=0.82, severity="Moderate")
        classification.all_scores = {
            "Normal": 0.10,
            "Pneumonia": 0.82,
            "Tuberculosis": 0.08,
        }
        report = {
            "findings": "Findings suspicious for tuberculosis.",
            "impression": "1. Consider tuberculosis.",
        }

        self.assertTrue(
            LLMReportEngine._is_report_supported(
                report, classification, allow_visual_details=True
            )
        )
        self.assertFalse(
            LLMReportEngine._is_report_supported(
                report, classification, allow_visual_details=False
            )
        )

    def test_json_parser_requires_grounded_clinical_sections(self):
        payload = {
            "technique": "Single image.",
            "comparison": "None.",
            "image_quality": "Limited.",
            "findings": "Visible structures assessed on the supplied image.",
            "impression": "1. Indeterminate finding.",
            "differential_diagnosis": "None.",
            "recommendations": "Review the complete study.",
            "critical_communication": "No critical communication generated.",
            "patient_explanation": "A concise explanation for the patient.",
        }
        parsed = LLMReportEngine._parse_json_report(json.dumps(payload))
        self.assertEqual(set(parsed), set(payload))

    def test_fallback_does_not_turn_confidence_severity_into_urgent_treatment(self):
        report = LLMReportEngine()._generate_template_report(result(), "brain_mri")
        combined = " ".join(report.values()).lower()
        self.assertNotIn("urgent", combined)
        self.assertNotIn("biopsy", combined)
        self.assertNotIn("surgery", combined)
        self.assertIn("single", report["technique"].lower())
        self.assertIn("insufficient", report["impression"].lower())

    def test_translation_chunks_stay_inside_api_limit(self):
        text = "Sentence for the patient. " * 250
        chunks = LLMReportEngine._chunk_translation_text(text, limit=1900)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 1900 for chunk in chunks))

    def test_brain_report_grounding_removes_unsupported_acquisition_claims(self):
        generated = {
            "technique": "Single exported coronal post-contrast MRI slice.",
            "comparison": "Compared with an unavailable prior examination.",
            "image_quality": "Diagnostic quality.",
            "findings": (
                "SELLA: A lobulated, heterogeneously enhancing sellar mass is visible, "
                "demonstrating concordance with the supportive classifier finding.\n"
                "BRAIN: Restricted diffusion is present."
            ),
            "impression": "1. Enhancing sellar mass, supported by the model.",
            "differential_diagnosis": "Pituitary adenoma.",
            "recommendations": "Obtain a complete pituitary MRI protocol.",
            "critical_communication": "No critical communication generated.",
            "patient_explanation": "",
        }

        report = LLMReportEngine._ground_report_to_available_input(
            generated, "brain_mri"
        )
        clinical_text = " ".join(
            report[field]
            for field in ("findings", "impression", "differential_diagnosis")
        ).lower()

        self.assertIn("pulse sequence", report["technique"].lower())
        self.assertIn("not available", report["technique"].lower())
        self.assertEqual(
            report["comparison"],
            "No prior imaging was supplied for comparison.",
        )
        self.assertIn("sellar mass", clinical_text)
        self.assertNotIn("enhanc", clinical_text)
        self.assertNotIn("classifier", clinical_text)
        self.assertNotIn("restricted diffusion", clinical_text)

    def test_grounding_preserves_anatomy_line_breaks(self):
        report = LLMReportEngine._ground_report_to_available_input(
            {
                "findings": "LUNGS/AIRWAYS: Clear.\nPLEURA: No visible pleural fluid.",
                "impression": "1. No focal chest abnormality identified.",
                "differential_diagnosis": "None based on the supplied image.",
            },
            "chest_xray",
        )
        self.assertIn("\n", report["findings"])


class PatientTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stored_patient_explanation_returns_without_an_extra_generation_call(self):
        engine = LLMReportEngine(gemini_api_key="configured")
        if engine.gemini_service:
            engine.gemini_service.generate_text = AsyncMock(return_value="should not be used")

        output = await engine._generate_patient_explanation_english({
            "patient_explanation": "A prepared plain-language explanation.",
            "scan_type": "chest_xray",
        })

        self.assertEqual(output, "A prepared plain-language explanation.")
        if engine.gemini_service:
            engine.gemini_service.generate_text.assert_not_awaited()

    async def test_existing_report_gets_an_immediate_grounded_english_explanation(self):
        engine = LLMReportEngine(gemini_api_key="configured")
        if engine.gemini_service:
            engine.gemini_service.generate_text = AsyncMock(return_value="should not be used")

        output = await engine._generate_patient_explanation_english({
            "scan_type": "brain_mri",
            "top_label": "Meningioma",
            "critical_communication": "No critical communication generated.",
        })

        self.assertIn("covering around the brain", output)
        self.assertIn("not a confirmed diagnosis", output)
        if engine.gemini_service:
            engine.gemini_service.generate_text.assert_not_awaited()

    async def test_patient_translation_uses_primary_translator_first(self):
        engine = LLMReportEngine(sarvam_api_key="configured", gemini_api_key="configured")
        engine._generate_patient_explanation_english = AsyncMock(return_value="English explanation")
        engine._translate_with_sarvam = AsyncMock(return_value="हिंदी विवरण")
        if engine.gemini_service:
            engine.gemini_service.generate_text = AsyncMock(return_value="fallback")

        output = await engine.generate_patient_report({}, "Hindi")

        self.assertEqual(output, "हिंदी विवरण")
        engine._translate_with_sarvam.assert_awaited_once_with("English explanation", "hi-IN")
        if engine.gemini_service:
            engine.gemini_service.generate_text.assert_not_awaited()

    async def test_patient_translation_falls_back_without_exposing_provider(self):
        engine = LLMReportEngine(sarvam_api_key="configured", gemini_api_key="configured")
        engine._generate_patient_explanation_english = AsyncMock(return_value="English explanation")
        engine._translate_with_sarvam = AsyncMock(return_value=None)
        if engine.gemini_service:
            engine.gemini_service.generate_text = AsyncMock(return_value="தமிழ் விளக்கம்")

        output = await engine.generate_patient_report({}, "Tamil")

        self.assertEqual(output, "தமிழ் விளக்கம்")
        self.assertNotIn("Gemini", output)
        self.assertNotIn("Sarvam", output)


class GeminiToNvidiaFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_hedged_report_switches_immediately_to_nvidia(self):
        from unittest.mock import MagicMock
        from PIL import Image

        engine = LLMReportEngine(gemini_api_key="configured", nvidia_api_key="configured")

        # Gemini returns a hedged report with no proper findings
        hedged_report_mock = MagicMock()
        hedged_report_mock.model_dump.return_value = {
            "technique": "Single frontal chest radiograph.",
            "comparison": "No prior examination supplied.",
            "image_quality": "Limited visual quality.",
            "findings": "Indeterminate chest radiograph, no definitive abnormality identified.",
            "impression": "1. Indeterminate finding. No acute abnormality.",
            "differential_diagnosis": "None.",
            "recommendations": "Clinical correlation.",
            "critical_communication": "No critical communication generated.",
            "patient_explanation": "Findings are unclear.",
        }
        engine.gemini_service = MagicMock()
        engine.gemini_service.generate_structured_report = AsyncMock(return_value=hedged_report_mock)

        # NVIDIA service returns an authoritative report
        nv_report = {
            "technique": "Single frontal chest radiograph.",
            "comparison": "No prior examination supplied.",
            "image_quality": "Adequate.",
            "findings": "LUNGS: Consolidation in the right lower lobe consistent with acute pneumonia. PLEURA: No effusion.",
            "impression": "1. Right lower lobe pneumonia.",
            "differential_diagnosis": "Bacterial pneumonia, viral pneumonia.",
            "recommendations": "Correlate clinically and monitor antibiotic response.",
            "critical_communication": "Routine notification.",
            "patient_explanation": "Signs of a lung infection were identified.",
        }
        engine.nvidia_service = MagicMock()
        engine.nvidia_service.generate_report = MagicMock(return_value=nv_report)
        engine.nvidia_service.verify_report = MagicMock(return_value=MagicMock(passes=True, recommendation="accept", issues=[]))

        res = result(label="Pneumonia", confidence=0.95, severity="Moderate")
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        report = await engine.generate_report(res, "chest_xray", image=img)

        # Gemini was called, but failed _has_proper_findings
        engine.gemini_service.generate_structured_report.assert_awaited_once()
        # Immediately fell back to NVIDIA
        engine.nvidia_service.generate_report.assert_called_once()
        self.assertIn("pneumonia", report["findings"].lower())
        self.assertIn("pneumonia", report["impression"].lower())

    async def test_gemini_rate_limited_switches_immediately_to_nvidia(self):
        from unittest.mock import MagicMock
        from PIL import Image

        engine = LLMReportEngine(gemini_api_key="configured", nvidia_api_key="configured")

        # Gemini raises rate limit / quota exceeded
        engine.gemini_service = MagicMock()
        engine.gemini_service.generate_structured_report = AsyncMock(side_effect=RuntimeError("429 ResourceExhausted: Quota exceeded"))

        nv_report = {
            "technique": "Single frontal chest radiograph.",
            "comparison": "No prior examination supplied.",
            "image_quality": "Adequate.",
            "findings": "LUNGS: Consolidation in the right lower lobe. PLEURA: Clear.",
            "impression": "1. Right lower lobe pneumonia.",
            "differential_diagnosis": "Bacterial pneumonia.",
            "recommendations": "Follow-up imaging.",
            "critical_communication": "Routine notification.",
            "patient_explanation": "Lung infection identified.",
        }
        engine.nvidia_service = MagicMock()
        engine.nvidia_service.generate_report = MagicMock(return_value=nv_report)
        engine.nvidia_service.verify_report = MagicMock(return_value=MagicMock(passes=True, recommendation="accept", issues=[]))

        res = result(label="Pneumonia", confidence=0.95, severity="Moderate")
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        report = await engine.generate_report(res, "chest_xray", image=img)

        engine.nvidia_service.generate_report.assert_called_once()
        self.assertIn("pneumonia", report["impression"].lower())

    def test_scan_type_verifier_gemini_failure_switches_immediately_to_nvidia(self):
        from unittest.mock import MagicMock
        from PIL import Image
        from services.scan_type_verifier import ScanTypeVerifier, ScanTypeVerification

        verifier = ScanTypeVerifier(
            api_key="test_gemini",
            model="gemini-flash-latest",
            nvidia_api_key="test_nvidia",
            nvidia_model="meta/llama-3.2-11b-vision-instruct",
        )
        # Mock local model returning None (deferring to vision models)
        verifier._verify_locally = MagicMock(return_value=None)
        # Mock Gemini failing
        verifier._verify_with_gemini = MagicMock(side_effect=RuntimeError("Gemini 429 Quota Exceeded"))
        # Mock NVIDIA succeeding
        nv_verif = ScanTypeVerification(
            category="chest_xray",
            confidence=0.98,
            is_single_diagnostic_image=True,
            anatomy_complete_enough=True,
            reason="Frontal chest radiograph showing thorax.",
        )
        verifier._verify_with_nvidia = MagicMock(return_value=nv_verif)

        img = Image.new("RGB", (256, 256), (120, 120, 120))
        result_verif = verifier._verify_sync(img)

        verifier._verify_with_gemini.assert_called_once()
        verifier._verify_with_nvidia.assert_called_once()
        self.assertEqual(result_verif.category, "chest_xray")


if __name__ == "__main__":
    unittest.main()
