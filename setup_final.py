import os
import glob

# 1. Update .gitignore
gitignore_path = '.gitignore'
with open(gitignore_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() in ['!models/best_brain_model.keras', '!models/cnn_lung_model.pth', '!models/cnn_Kidney_Stone_model.pth']:
        continue
    new_lines.append(line)

new_lines.append('\n# Active Jeevansh models\n!models/jeevansh/\n!models/jeevansh/*\n')

with open(gitignore_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# 2. Create .env
with open('.env.example', 'r', encoding='utf-8') as f:
    env_content = f.read()

env_content = env_content.replace('MedRittAI', 'MedRittAI').replace('medrittai-change-this-in-production', 'medrittai-change-this-in-production')

with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

# 3. Delete e* file
for f in glob.glob('e*'):
    if 'e\xef\x80\xa2' in f or len(f) <= 2:
        try:
            os.remove(f)
        except:
            pass

print('Setup script complete.')
