"""Regression tests for native professional PDF generation."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.pdf_generator import PDFGenerator


class PDFGeneratorTests(unittest.TestCase):
    def test_professional_pdf_is_multipage_and_includes_heatmap(self):
        repeated = (
            "LUNGS/AIRWAYS: Patchy opacity is described on the supplied image.\n"
            "PLEURA: No supported pleural abnormality is described.\n"
            "CARDIOMEDIASTINAL SILHOUETTE: Limited assessment on the supplied image. "
        ) * 8
        report = {
            "patient_id": "TEST-001",
            "scan_date": "2026-07-23",
            "scan_type": "chest_xray",
            "modality": "X-ray",
            "top_label": "Pneumonia",
            "confidence": 0.87,
            "severity": "Severe",
            "clinical_history": "Cough and fever; duration not provided.",
            "technique": "Single frontal chest radiograph.",
            "comparison": "No prior imaging supplied.",
            "image_quality": "Adequate for limited automated review.",
            "findings": repeated,
            "impression": "1. Airspace opacity suspicious for an infectious process.",
            "differential_diagnosis": "Atelectatic change remains a consideration.",
            "recommendations": "Correlate with symptoms and direct image review.",
            "critical_communication": "No critical communication generated.",
            "all_scores": {"Pneumonia": 0.87, "Atelectasis": 0.32, "No Finding": 0.04},
            "heatmap_target_label": "Pneumonia",
            "methodology": "Classification with Grad-CAM heatmap explainability.",
            "limitations": "Performance is limited to trained categories.",
            "disclaimer": "Clinician verification is required before clinical use.",
            "doctor_assessment": "Reviewed and approved for the patient record.",
            "is_final": True,
        }

        with tempfile.TemporaryDirectory() as directory:
            heatmap = Path(directory) / "heatmap.png"
            Image.new("RGB", (512, 512), (120, 45, 30)).save(heatmap)
            pdf = PDFGenerator().generate_pdf(report, "e5d7859e-test", str(heatmap))

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 5_000)
        self.assertGreaterEqual(pdf.count(b"/Type /Page"), 2)

    def test_professional_pdf_dual_images_for_all_tests(self):
        test_cases = [
            ("skin_cancer", "classification", "Vascular_Lesion"),
            ("pneumonia", "classification", "Pneumonia"),
            ("brain_tumor", "classification", "Pituitary"),
            ("bone_fracture", "detection", "Fracture"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            heatmap_path = Path(directory) / "heatmap.png"
            orig_path = Path(directory) / "orig.png"
            Image.new("RGB", (256, 256), (255, 100, 0)).save(heatmap_path)
            Image.new("RGB", (256, 256), (50, 50, 50)).save(orig_path)

            for scan_type, task_type, label in test_cases:
                report = {
                    "patient_id": "TEST-DUAL",
                    "scan_date": "2026-10-04",
                    "scan_type": scan_type,
                    "task_type": task_type,
                    "modality": "X-ray" if "xray" in scan_type or "fracture" in scan_type or "pneumonia" in scan_type else "Dermatoscopic",
                    "top_label": label,
                    "confidence": 0.95,
                    "severity": "Moderate",
                    "clinical_history": "Clinical test history.",
                    "technique": "Diagnostic acquisition.",
                    "findings": "Clear visualization of region.",
                    "impression": f"Identified {label}.",
                    "all_scores": {label: 0.95, "Negative": 0.05},
                    "heatmap_target_label": label,
                    "is_final": True,
                }
                pdf = PDFGenerator().generate_pdf(
                    report,
                    f"test-{scan_type}",
                    heatmap_path=str(heatmap_path),
                    original_image_path=str(orig_path),
                )
                self.assertTrue(pdf.startswith(b"%PDF"))
                self.assertGreater(len(pdf), 5_000)


if __name__ == "__main__":
    unittest.main()
