import os
import re

replacements = [
    (
        r"C:\\MedRittAI\\frontend\\src\\components\\BrandLogo.tsx",
        [
            ("import medrittLogo from '../../../medritt_logo-removebg-preview.png';", "import brandLogo from '../../../MedRittAI_Healthcare_Logo-removebg-preview.png';"),
            ("import medrittLoginLogo from '../../../medritt_logo.jpeg';", ""),
            ("src={isLoginLogo ? medrittLoginLogo : medrittLogo}", "src={brandLogo}"),
            ("alt=\"MedRittAI logo\"", "alt=\"MedRittAI logo\"")
        ]
    ),
    (
        r"C:\\MedRittAI\\frontend\\src\\api\\client.ts",
        [("MedRittAI_Report_", "MedRittAI_Report_")]
    ),
    (
        r"C:\\MedRittAI\\frontend\\public\\site.webmanifest",
        [("MedRittAI", "MedRittAI")]
    ),
    (
        r"C:\\MedRittAI\\frontend\\index.html",
        [("MedRittAI", "MedRittAI"), ("MedRitt clinical", "MedRittAI clinical")]
    ),
    (
        r"C:\\MedRittAI\\frontend\\src\\pages\\LandingPage.tsx",
        [("MedRittAI", "MedRittAI")]
    ),
    (
        r"C:\\MedRittAI\\backend\\services\\pdf_generator.py",
        [
            ('title=f"MedRittAI', 'title=f"MedRittAI'),
            ('author="MedRittAI"', 'author="MedRittAI"'),
            ('"MEDRITTAI  /  CLINICAL IMAGING"', '"MEDRITTAI  /  CLINICAL IMAGING"'),
            ('Paragraph("MedRittAI · Complete Case Study"', 'Paragraph("MedRittAI · Complete Case Study"'),
            ('"MedRittAI Final Clinical', '"MedRittAI Final Clinical'),
            ('"MedRittAI Preliminary', '"MedRittAI Preliminary')
        ]
    ),
    (
        r"C:\\MedRittAI\\backend\\services\\llm_report_engine.py",
        [("MedRittAI", "MedRittAI")]
    ),
    (
        r"C:\\MedRittAI\\backend\\templates\\report.html",
        [("MedRittAI", "MedRittAI")]
    ),
    (
        r"C:\\MedRittAI\\backend\\templates\\report.txt",
        [("MEDRITTAI", "MEDRITTAI")]
    ),
    (
        r"C:\\MedRittAI\\backend\\routers\\report.py",
        [("MedRittAI_Report_", "MedRittAI_Report_")]
    ),
    (
        r"C:\\MedRittAI\\backend\\routers\\case_study.py",
        [("MedRittAI_Case_", "MedRittAI_Case_")]
    ),
    (
        r"C:\\MedRittAI\\backend\\config.py",
        [('APP_NAME: str = "MedRittAI"', 'APP_NAME: str = "MedRittAI"')]
    ),
    (
        r"C:\\MedRittAI\\backend\\main.py",
        [
            ("MedRittAI Diagnostic", "MedRittAI Diagnostic"),
            ("MedRittAI backend", "MedRittAI backend"),
            ("MedRittAI shutting", "MedRittAI shutting"),
            ("MedRitt Administrator", "MedRitt Administrator"),
            ("MedRitt Care Pharmacy", "MedRitt Care Pharmacy"),
            ("MedRitt Hospital", "MedRitt Hospital"),
            ("pharmacy@medritt.local", "pharmacy@medritt.local"),
            ('title="MedRittAI API"', 'title="MedRittAI API"')
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
