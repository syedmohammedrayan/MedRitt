# MedRittAI Backend

FastAPI backend for authentication, uploads, ML inference, model attribution, LLM-assisted reports, and PDF export.

## Setup

From the repo root:

```powershell
cd backend
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Model Configuration

The backend uses the Jeevansh model engine. All models are initialized from the local `models/jeevansh/` directory via the unified `ModelRegistry`.

Currently active models:
- skin_cancer (`skin_cancer.pth`)
- pneumonia (`pneumonia.pth`)
- brain_tumor (`brain_tumour.pt`)
- bone_fracture (`fracture.pt`)

## Run

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected model status:

```text
skin_cancer: loaded
pneumonia: loaded
brain_tumor: loaded
bone_fracture: loaded
```

## Runtime Data

The backend creates runtime data under:

```text
data/uploads
data/heatmaps
data/thumbnails
```

The SQLite database is:

```text
data/app.db
```
