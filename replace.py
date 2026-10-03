import os
import re

replacements = [
    (
        r"C:\\MedoraAI\\frontend\\src\\components\\BrandLogo.tsx",
        [
            ("import medoraLogo from '../../../medora_logo-removebg-preview.png';", "import brandLogo from '../../../MedRittAI_Healthcare_Logo-removebg-preview.png';"),
            ("import medoraLoginLogo from '../../../medora_logo.jpeg';", ""),
            ("src={isLoginLogo ? medoraLoginLogo : medoraLogo}", "src={brandLogo}"),
            ("alt=\"MedoraAI logo\"", "alt=\"MedRittAI logo\"")
        ]
    ),
    (
        r"C:\\MedoraAI\\frontend\\src\\api\\client.ts",
        [("MedoraAI_Report_", "MedRittAI_Report_")]
    ),
    (
        r"C:\\MedoraAI\\frontend\\public\\site.webmanifest",
        [("MedoraAI", "MedRittAI")]
    ),
    (
        r"C:\\MedoraAI\\frontend\\index.html",
        [("MedoraAI", "MedRittAI"), ("Medora clinical", "MedRittAI clinical")]
    ),
    (
        r"C:\\MedoraAI\\frontend\\src\\pages\\LandingPage.tsx",
        [("MedoraAI", "MedRittAI")]
    ),
    (
        r"C:\\MedoraAI\\backend\\services\\pdf_generator.py",
        [
            ('title=f"MedoraAI', 'title=f"MedRittAI'),
            ('author="MedoraAI"', 'author="MedRittAI"'),
            ('"MEDORAAI  /  CLINICAL IMAGING"', '"MEDRITTAI  /  CLINICAL IMAGING"'),
            ('Paragraph("MedoraAI · Complete Case Study"', 'Paragraph("MedRittAI · Complete Case Study"'),
            ('"MedoraAI Final Clinical', '"MedRittAI Final Clinical'),
            ('"MedoraAI Preliminary', '"MedRittAI Preliminary')
        ]
    ),
    (
        r"C:\\MedoraAI\\backend\\services\\llm_report_engine.py",
        [("MedoraAI", "MedRittAI")]
    ),
    (
        r"C:\\MedoraAI\\backend\\templates\\report.html",
        [("MedoraAI", "MedRittAI")]
    ),
    (
        r"C:\\MedoraAI\\backend\\templates\\report.txt",
        [("MEDORAAI", "MEDRITTAI")]
    ),
    (
        r"C:\\MedoraAI\\backend\\routers\\report.py",
        [("MedoraAI_Report_", "MedRittAI_Report_")]
    ),
    (
        r"C:\\MedoraAI\\backend\\routers\\case_study.py",
        [("MedoraAI_Case_", "MedRittAI_Case_")]
    ),
    (
        r"C:\\MedoraAI\\backend\\config.py",
        [('APP_NAME: str = "MedoraAI"', 'APP_NAME: str = "MedRittAI"')]
    ),
    (
        r"C:\\MedoraAI\\backend\\main.py",
        [
            ("MedoraAI Diagnostic", "MedRittAI Diagnostic"),
            ("MedoraAI backend", "MedRittAI backend"),
            ("MedoraAI shutting", "MedRittAI shutting"),
            ("Medora Administrator", "MedRitt Administrator"),
            ("Medora Care Pharmacy", "MedRitt Care Pharmacy"),
            ("Medora Hospital", "MedRitt Hospital"),
            ("pharmacy@medora.local", "pharmacy@medritt.local"),
            ('title="MedoraAI API"', 'title="MedRittAI API"')
        ]
    )
]

for filepath, reps in replacements:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in reps:
        new_content = new_content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("Replacements complete.")
