"""Phase 16 orchestrator tests A-T."""
from __future__ import annotations
import asyncio, json, unittest, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from services.ai_orchestrator import (
    MedRittOrchestrator, ProviderStatus, ReportDecision,
    ReportPipelineResult, ScanPipelineResult,
)


# ---- helpers ---------------------------------------------------------------

def _make_orchestrator(report_engine=None, scan_verifier=None, nvidia_service=None, cloudinary_enabled=False):
    return MedRittOrchestrator(
        report_engine=report_engine or _mock_report_engine(),
        scan_verifier=scan_verifier or _mock_scan_verifier(),
        nvidia_service=nvidia_service,
        cloudinary_enabled=cloudinary_enabled,
    )

def _mock_scan_verifier(sane=True):
    from services.scan_type_verifier import ScanTypeVerification
    m = MagicMock()
    m.verify = AsyncMock(return_value=ScanTypeVerification(
        category="chest_xray", confidence=0.95,
        is_single_diagnostic_image=sane, anatomy_complete_enough=sane,
        reason="mock ok",
    ))
    return m

def _mock_report_engine(provider="gemini", raises=False):
    m = MagicMock()
    if raises:
        m.generate_report = AsyncMock(side_effect=RuntimeError("Gemini down"))
    else:
        m.generate_report = AsyncMock(return_value={"llm_provider": provider, "findings": "ok", "impression": "ok"})
    m._generate_template_report = MagicMock(return_value={"llm_provider": "template", "findings": "tmpl", "impression": "tmpl"})
    return m

def _mock_nvidia(passes=True, recommendation="accept"):
    from services.nvidia_service import ReportQAResult
    r = ReportQAResult(passes=passes, issues=[] if passes else ["issue"],
                       unsupported_claims=[], grounding_warnings=[], recommendation=recommendation)
    m = MagicMock()
    m.verify_report = MagicMock(return_value=r)
    return m

@dataclass
class _JR:
    task_type: str = "classification"
    top_label: str = "Normal"
    confidence: float = 0.92
    all_scores: dict = field(default_factory=lambda: {"Normal": 0.92})
    severity: Optional[str] = "Low"
    bounding_boxes: list = field(default_factory=list)
    is_low_confidence: bool = False
    heatmap_target_label: str = "Normal"
    secondary_findings: list = field(default_factory=list)

_CL = {"task_type": "classification", "top_label": "Normal", "confidence": 0.92,
       "all_scores": {"Normal": 0.92}, "severity": "Low"}
_DT = {"task_type": "detection", "top_label": "Fracture", "confidence": 0.88,
       "detections": [{"class": "Fracture", "confidence": 0.88}]}
_IMG = MagicMock()


# ---- tests -----------------------------------------------------------------

class TestA_FullScan(unittest.TestCase):
    def test_returns_jeevansh_fields(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="aaa", jeevansh_result=_CL))
        self.assertEqual(r.top_label, "Normal")
        self.assertAlmostEqual(r.confidence, 0.92)
        self.assertIsNone(r.error)

class TestB_Classification(unittest.TestCase):
    def test_preserves_all_scores_and_severity(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="bbb", jeevansh_result=_CL))
        self.assertIsNotNone(r.all_scores)
        self.assertEqual(r.severity, "Low")

class TestC_Detection(unittest.TestCase):
    def test_null_all_scores_and_severity(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="bone_fracture", scan_id="ccc", jeevansh_result=_DT))
        self.assertIsNone(r.all_scores)
        self.assertIsNone(r.severity)
        self.assertEqual(r.task_type, "detection")

class TestD_GroqUnavailable(unittest.TestCase):
    def test_jeevansh_survives_groq_failure(self):
        sv = MagicMock(); sv.verify = AsyncMock(side_effect=RuntimeError("groq down"))
        r = asyncio.run(_make_orchestrator(scan_verifier=sv).run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="ddd", jeevansh_result=_CL))
        self.assertEqual(r.providers["groq"]["status"], ProviderStatus.UNAVAILABLE)
        self.assertEqual(r.top_label, "Normal")

class TestE_GeminiUnavailable(unittest.TestCase):
    def test_template_fallback(self):
        r = asyncio.run(_make_orchestrator(report_engine=_mock_report_engine(raises=True)).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r.llm_provider, "template")
        self.assertEqual(r.decision, ReportDecision.FALLBACK_TEMPLATE)

class TestF_NVIDIAUnavailable(unittest.TestCase):
    def test_gemini_report_retained(self):
        nv = MagicMock(); nv.verify_report = MagicMock(return_value=None)
        r = asyncio.run(_make_orchestrator(nvidia_service=nv).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r.llm_provider, "gemini")
        self.assertEqual(r.decision, ReportDecision.ACCEPTED_NO_QA)
        self.assertEqual(r.providers["nvidia"]["status"], ProviderStatus.UNAVAILABLE)

class TestG_NVIDIAPass(unittest.TestCase):
    def test_accepted(self):
        r = asyncio.run(_make_orchestrator(nvidia_service=_mock_nvidia(passes=True)).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r.decision, ReportDecision.ACCEPTED)
        self.assertTrue(r.nvidia_qa["passes"])

class TestH_NVIDIAFlag(unittest.TestCase):
    def test_template_fallback_on_flag(self):
        r = asyncio.run(_make_orchestrator(nvidia_service=_mock_nvidia(passes=False, recommendation="flag")).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r.decision, ReportDecision.FALLBACK_TEMPLATE)
        self.assertEqual(r.llm_provider, "template")

class TestI_JeevanshFailure(unittest.TestCase):
    def test_no_manufactured_diagnosis(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="iii", jeevansh_result={}))
        self.assertIsNone(r.top_label)
        self.assertIsNone(r.confidence)

class TestJ_CloudinaryFailure(unittest.TestCase):
    def test_no_fake_url(self):
        orch = _make_orchestrator(cloudinary_enabled=True)
        with patch("services.cloudinary_storage.upload_original_scan", side_effect=RuntimeError("cdn down")):
            r = asyncio.run(orch.run_scan_pipeline(
                image=_IMG, scan_type="chest_xray", scan_id="jjj",
                jeevansh_result=_CL, original_image_buffer=b"bytes"))
        self.assertIsNone(r.original_image_url)
        self.assertEqual(r.top_label, "Normal")

class TestK_SQLiteStructure(unittest.TestCase):
    def test_correlation_id_is_uuid(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="kkk", jeevansh_result=_CL))
        self.assertRegex(r.correlation_id, r"[0-9a-f-]{36}")

class TestL_NoPDFOnReport(unittest.TestCase):
    def test_run_report_pipeline_does_not_generate_pdf(self):
        r = asyncio.run(_make_orchestrator().run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertFalse(hasattr(r, "pdf_url"))

class TestM_GradcamYolo(unittest.TestCase):
    def test_gradcam_upload_for_classification(self):
        orch = _make_orchestrator(cloudinary_enabled=True)
        with patch("services.cloudinary_storage.upload_gradcam", return_value=("https://cdn/hm.png", "pub_hm")) as mu:
            r = asyncio.run(orch.run_scan_pipeline(
                image=_IMG, scan_type="chest_xray", scan_id="mmm1",
                jeevansh_result=_CL, heatmap_array=MagicMock()))
        mu.assert_called_once()
        self.assertEqual(r.heatmap_url, "https://cdn/hm.png")

    def test_yolo_upload_for_detection(self):
        orch = _make_orchestrator(cloudinary_enabled=True)
        with patch("services.cloudinary_storage.upload_yolo_overlay", return_value=("https://cdn/yolo.png", "pub_yolo")) as mu:
            r = asyncio.run(orch.run_scan_pipeline(
                image=_IMG, scan_type="bone_fracture", scan_id="mmm2",
                jeevansh_result=_DT, overlay_array=MagicMock()))
        mu.assert_called_once()
        self.assertEqual(r.overlay_url, "https://cdn/yolo.png")

class TestN_ReportDataSerializable(unittest.TestCase):
    def test_report_dict_is_json_serializable(self):
        r = asyncio.run(_make_orchestrator().run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertIsInstance(json.dumps(r.report_data), str)

class TestO_Idempotent(unittest.TestCase):
    def test_repeated_calls_consistent(self):
        orch = _make_orchestrator()
        r1 = asyncio.run(orch.run_report_pipeline(jeevansh_result=_JR(), scan_type="chest_xray"))
        r2 = asyncio.run(orch.run_report_pipeline(jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r1.llm_provider, r2.llm_provider)

class TestP_InvalidJSON(unittest.TestCase):
    def test_malformed_report_falls_back_to_template(self):
        re = _mock_report_engine(raises=True)
        r = asyncio.run(_make_orchestrator(report_engine=re).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertEqual(r.llm_provider, "template")

class TestQ_MarkdownFencedNVIDIA(unittest.TestCase):
    def test_strips_fences(self):
        fenced = "`json\n{\"passes\": true, \"issues\": [], \"unsupported_claims\": [], \"grounding_warnings\": [], \"recommendation\": \"accept\"}\n`"
        cleaned = fenced.strip()
        if cleaned.startswith("`"):
            lines = cleaned.splitlines()
            lines = lines[1:] if lines[0].startswith("`") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("`") else lines
            cleaned = "\n".join(lines).strip()
        self.assertTrue(json.loads(cleaned)["passes"])

class TestR_Timeout(unittest.TestCase):
    def test_groq_timeout_does_not_hang(self):
        async def slow(*a, **k): await asyncio.sleep(60)
        sv = MagicMock(); sv.verify = slow
        t0 = time.perf_counter()
        r = asyncio.run(_make_orchestrator(scan_verifier=sv).run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="rrr", jeevansh_result=_CL))
        self.assertLess(time.perf_counter() - t0, 40.0)
        self.assertEqual(r.top_label, "Normal")

class TestS_NoSecrets(unittest.TestCase):
    def _check(self, d):
        s = json.dumps(d, default=str)
        for term in ["password", "SECRET_KEY", "api_key", "API_SECRET", "CLOUDINARY_API_SECRET"]:
            self.assertNotIn(term, s, f"Secret '{term}' leaked")
    def test_scan_result(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="sss", jeevansh_result=_CL))
        self._check(r.__dict__)
    def test_report_result(self):
        r = asyncio.run(_make_orchestrator().run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self._check(r.__dict__)

class TestT_NoDiagnosticOverride(unittest.TestCase):
    def _assert_unchanged(self, r):
        self.assertEqual(r.top_label, "Normal")
        self.assertAlmostEqual(r.confidence, 0.92)

    def test_gemini_cannot_overwrite_jeevansh(self):
        re = MagicMock()
        re.generate_report = AsyncMock(return_value={"llm_provider": "gemini", "top_label": "INJECTED", "confidence": 0.0})
        re._generate_template_report = MagicMock(return_value={"llm_provider": "template"})
        r = asyncio.run(_make_orchestrator(report_engine=re).run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="t1", jeevansh_result=_CL))
        self._assert_unchanged(r)

    def test_groq_cannot_overwrite_jeevansh(self):
        from services.scan_type_verifier import ScanTypeVerification
        sv = MagicMock()
        sv.verify = AsyncMock(return_value=ScanTypeVerification(
            category="other", confidence=0.99, is_single_diagnostic_image=True,
            anatomy_complete_enough=False, reason="groq says wrong"))
        r = asyncio.run(_make_orchestrator(scan_verifier=sv).run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="t2", jeevansh_result=_CL))
        self._assert_unchanged(r)

    def test_nvidia_cannot_overwrite_jeevansh(self):
        nv = _mock_nvidia(passes=False, recommendation="flag")
        r = asyncio.run(_make_orchestrator(nvidia_service=nv).run_report_pipeline(
            jeevansh_result=_JR(), scan_type="chest_xray"))
        self.assertTrue(r.providers["jeevansh"]["authoritative"])
        self.assertFalse(hasattr(r, "top_label"))

    def test_orchestrator_cannot_manufacture_diagnosis(self):
        r = asyncio.run(_make_orchestrator().run_scan_pipeline(
            image=_IMG, scan_type="chest_xray", scan_id="t4", jeevansh_result={}))
        self.assertIsNone(r.top_label)
        self.assertIsNone(r.confidence)
        self.assertIsNone(r.severity)

if __name__ == "__main__":
    unittest.main()
