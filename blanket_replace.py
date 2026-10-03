import os

replacements = [
    ('MedRittAI', 'MedRittAI'),
    ('medrittai', 'medrittai'),
    ('MEDRITTAI', 'MEDRITTAI'),
    ('MedRitt', 'MedRitt'),
    ('medritt', 'medritt'),
    ('MEDRITT', 'MEDRITT')
]

exclude_dirs = {'.git', '.git-old', 'node_modules', '__pycache__', 'dist', '.vite', '.agents', '.gemini', '.agent'}
exclude_exts = {'.png', '.jpg', '.jpeg', '.pdf', '.pt', '.pth', '.keras', '.h5', '.db', '.db-journal', '.sqlite3', '.zip', '.svg', '.ico', '.pyc'}

count = 0
for root, dirs, files in os.walk('C:\\MedRitt'):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in exclude_exts:
            continue
            
        filepath = os.path.join(root, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue # skip binary or non-utf8 files
            
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Updated {filepath}')
            count += 1

print(f'Total files updated: {count}')
