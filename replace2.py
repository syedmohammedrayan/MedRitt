import os
import re

replacements = [
    (
        r"C:\\MedRittAI\\frontend\\src\\pages\\LoginPage.tsx",
        [("MedRitt Clinical Workspace", "MedRittAI Clinical Workspace")]
    ),
    (
        r"C:\\MedRittAI\\frontend\\src\\pages\\PharmacyBillPage.tsx",
        [
            ("MedRitt clinician", "MedRitt clinician"),
            ("MedRittAI", "MedRittAI")
        ]
    ),
    (
        r"C:\\MedRittAI\\frontend\\src\\pages\\RegisterPage.tsx",
        [("MedRittAI", "MedRittAI")]
    ),
    (
        r"C:\\MedRittAI\\frontend\\src\\pages\\CaseStudyView.tsx",
        [
            ("MedRittAI_Case_", "MedRittAI_Case_"),
            ("MedRittAI Hospital Intelligence", "MedRittAI Hospital Intelligence")
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
