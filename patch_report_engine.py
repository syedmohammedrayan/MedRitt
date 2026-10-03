import sys
import re

def patch():
    file_path = r"C:\MedoraAI\backend\services\llm_report_engine.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch _build_user_prompt
    build_user_prompt_replacement = """        elif scan_type == "pneumonia":
            scores_text = "\\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
                if score >= 0.05
            ) or "  - No secondary score reached the reporting threshold."
            return f\"\"\"EXAM: Chest radiograph
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

Generate a detailed, grounded chest radiograph report for clinical review.\"\"\"

        elif scan_type == "skin_cancer":
            scores_text = "\\n".join(
                f"  - {label}: {score * 100:.1f}%"
                for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1])
            )
            return f\"\"\"EXAM: Dermatoscopic image
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

Generate a detailed, grounded dermatoscopic report for clinical review.\"\"\"

        elif getattr(result, "task_type", None) == "detection" or scan_type in {"brain_tumor", "bone_fracture"}:
            exam_name = "Brain MRI image" if scan_type == "brain_tumor" else "Bone X-ray image"
            bboxes = getattr(result, "bounding_boxes", [])
            if bboxes:
                det_text = "\\n".join(f"  - {d['class']} (Confidence: {d['confidence']*100:.1f}%)" for d in bboxes)
            else:
                det_text = "  - 0 detections."
            return f\"\"\"EXAM: {exam_name}
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

Generate a detailed, grounded report for clinical review.\"\"\"

        else:"""
    content = content.replace("        else:\n            return f\"\"\"SCAN TYPE: Medical Image ({scan_type})", build_user_prompt_replacement + '\n            return f"""SCAN TYPE: Medical Image ({scan_type})')

    # 2. Patch generate_report methodology
    methodology_replacement = """        elif scan_type == "skin_cancer":
            methodology = "Classification was performed using a MobileNetV3-Large network for 7-class skin lesion classification. The accompanying heatmap uses Grad-CAM for localization."
        elif scan_type == "pneumonia":
            methodology = "Classification was performed using a MobileNetV3-Large network. The accompanying heatmap uses Grad-CAM for localization."
        elif scan_type == "brain_tumor":
            methodology = "Detection was performed using a YOLOv9m model. The accompanying image is a bounding-box overlay."
        elif scan_type == "bone_fracture":
            methodology = "Detection was performed using a YOLO11 model. The accompanying image is a bounding-box overlay."
        else:"""
    content = content.replace("        else:\n            methodology = \"AI model classification with model-attribution explainability.\"", methodology_replacement + '\n            methodology = "AI model classification with model-attribution explainability."')

    # 3. Patch _generate_template_report
    template_report_add = """        elif scan_type in {"pneumonia", "skin_cancer"}:
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
"""
    content = content.replace("        else:\n            technique = \"Single medical image submitted for limited review.\"", template_report_add + '        else:\n            technique = "Single medical image submitted for limited review."')

    # 4. Patch _complete_report_sections
    complete_report_add = """            "pneumonia": "Single chest radiograph submitted.",
            "skin_cancer": "Single dermatoscopic image submitted.",
            "brain_tumor": "Single brain MRI image submitted.",
            "bone_fracture": "Single bone X-ray image submitted.",
"""
    content = content.replace("            \"kidney_us\": \"Single kidney ultrasound image submitted; acquisition details and complete study are not provided.\",", "            \"kidney_us\": \"Single kidney ultrasound image submitted; acquisition details and complete study are not provided.\",\n" + complete_report_add)

    # 5. Patch _ground_report_to_available_input
    ground_report_add = """        elif scan_type in {"pneumonia", "skin_cancer", "brain_tumor", "bone_fracture"}:
            grounded["technique"] = f"Single uploaded {scan_type.replace('_', ' ')} image."
            grounded["image_quality"] = "Limited diagnostic assessment."
"""
    content = content.replace("            grounded[\"image_quality\"] = (\n                \"Limited diagnostic assessment because only one exported image is available; \"\n                \"subtle and out-of-frame findings cannot be excluded.\"\n            )", "            grounded[\"image_quality\"] = (\n                \"Limited diagnostic assessment because only one exported image is available; \"\n                \"subtle and out-of-frame findings cannot be excluded.\"\n            )\n" + ground_report_add)

    # 6. Patch _generate_patient_explanation_english
    patient_report_add = """            "pneumonia": "chest X-ray",
            "skin_cancer": "dermatoscopy image",
            "brain_tumor": "brain MRI image",
            "bone_fracture": "bone X-ray",
"""
    content = content.replace("            \"kidney_us\": \"kidney ultrasound image\",", "            \"kidney_us\": \"kidney ultrasound image\",\n" + patient_report_add)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    patch()
