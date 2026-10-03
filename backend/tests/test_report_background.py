import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi import BackgroundTasks
from PIL import Image

from routers.scan import _store_generated_report, analyze_scan


class BackgroundReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_response_does_not_wait_for_report_completion(self):
        report_started = asyncio.Event()
        report_release = asyncio.Event()

        async def generate_report(**_kwargs):
            report_started.set()
            await report_release.wait()
            return {"llm_provider": "gemini"}

        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "task_type": "classification",
            "top_label": "Glioma",
            "confidence": 0.98,
            "all_scores": {"Glioma": 0.98, "No Tumor": 0.02},
            "gradcam": np.zeros((32, 32, 3), dtype=np.uint8)
        }
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = mock_model

        scan = SimpleNamespace(
            id="scan-123",
            user_id=7,
            scan_type="brain_tumor",
            modality="MRI",
            file_path="",
            original_image_url=None,
            original_image_public_id=None,
            heatmap_url=None,
            heatmap_public_id=None,
        )
        from services.ai_orchestrator import ScanPipelineResult, ReportPipelineResult

        async def run_scan_pipeline(**kwargs):
            return ScanPipelineResult(
                top_label="Glioma",
                confidence=0.98,
                severity="High",
                all_scores={"Glioma": 0.98, "No Tumor": 0.02},
                bounding_boxes=[],
                heatmap_url=None,
                heatmap_public_id=None,
                overlay_url=None,
                overlay_public_id=None
            )

        async def run_report_pipeline(**kwargs):
            report_started.set()
            await report_release.wait()
            return ReportPipelineResult(
                report_data={"llm_provider": "gemini"},
                llm_provider="gemini",
                decision="accepted"
            )

        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    orchestrator=SimpleNamespace(
                        run_scan_pipeline=run_scan_pipeline,
                        run_report_pipeline=run_report_pipeline
                    ),
                    jeevansh_registry=mock_registry
                )
            )
        )
        background_tasks = BackgroundTasks()

        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            image_path = Path(temp_dir) / "scan.png"
            Image.new("RGB", (32, 32), "black").save(image_path)
            scan.file_path = str(image_path)

            with (
                patch("routers.scan.crud.get_scan", return_value=scan),
                patch("routers.scan.crud.update_scan_status"),
                patch("routers.scan.crud.update_scan_heatmap"),
                patch("routers.scan.crud.create_result"),
                patch("routers.scan.Image.fromarray") as fromarray,
            ):
                response = await analyze_scan(
                    scan_id=scan.id,
                    request=request,
                    background_tasks=background_tasks,
                    db=MagicMock(),
                    current_user=SimpleNamespace(id=7),
                )

        await asyncio.wait_for(report_started.wait(), timeout=1.0)
        self.assertEqual(response.scan_id, scan.id)
        self.assertEqual(response.status, "analyzed")
        self.assertEqual(len(background_tasks.tasks), 1)

        report_task = background_tasks.tasks[0].args[0]
        self.assertFalse(report_task.done())
        report_task.cancel()
        await asyncio.gather(report_task, return_exceptions=True)

    @patch("routers.scan.crud.replace_report")
    @patch("routers.scan.crud.get_scan")
    @patch("routers.scan.get_session_factory")
    async def test_completed_report_is_stored_with_an_independent_session(
        self,
        get_session_factory,
        get_scan,
        replace_report,
    ):
        report_db = MagicMock()
        get_session_factory.return_value.return_value = report_db
        get_scan.return_value = SimpleNamespace(id="scan-123")
        report_data = {"llm_provider": "gemini", "findings": "Grounded finding."}
        from services.ai_orchestrator import ReportPipelineResult
        pipeline_result = ReportPipelineResult(
            report_data=report_data,
            llm_provider="gemini",
            decision="accepted",
        )
        report_task = AsyncMock(return_value=pipeline_result)()

        await _store_generated_report(report_task, "scan-123")

        replace_report.assert_called_once_with(
            db=report_db,
            scan_id="scan-123",
            report_data=report_data,
            llm_provider="gemini",
        )
        report_db.close.assert_called_once_with()

    @patch("routers.scan.crud.replace_report")
    @patch("routers.scan.crud.get_scan")
    @patch("routers.scan.get_session_factory")
    async def test_report_is_discarded_if_scan_was_deleted(
        self,
        get_session_factory,
        get_scan,
        replace_report,
    ):
        report_db = MagicMock()
        get_session_factory.return_value.return_value = report_db
        get_scan.return_value = None
        from services.ai_orchestrator import ReportPipelineResult
        pipeline_result = ReportPipelineResult(
            report_data={"llm_provider": "gemini"},
            llm_provider="gemini",
            decision="accepted",
        )
        report_task = AsyncMock(return_value=pipeline_result)()

        await _store_generated_report(report_task, "deleted-scan")

        replace_report.assert_not_called()
        report_db.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
