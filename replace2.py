import os
import re

replacements = [
    (
        r"C:\\MedoraAI\\frontend\\src\\pages\\LoginPage.tsx",
        [("Medora Clinical Workspace", "MedRittAI Clinical Workspace")]
    ),
    (
        r"C:\\MedoraAI\\frontend\\src\\pages\\PharmacyBillPage.tsx",
        [
            ("Medora clinician", "MedRitt clinician"),
            ("MedoraAI", "MedRittAI")
        ]
    ),
    (
        r"C:\\MedoraAI\\frontend\\src\\pages\\RegisterPage.tsx",
        [("MedoraAI", "MedRittAI")]
    ),
    (
        r"C:\\MedoraAI\\frontend\\src\\pages\\CaseStudyView.tsx",
        [
            ("MedoraAI_Case_", "MedRittAI_Case_"),
            ("MedoraAI Hospital Intelligence", "MedRittAI Hospital Intelligence")
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

print("Second replacements complete.")
