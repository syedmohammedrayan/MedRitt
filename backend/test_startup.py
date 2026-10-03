import sys
sys.path.append('.')
from fastapi.testclient import TestClient
from main import app

try:
    with TestClient(app) as client:
        print('---HEALTH---')
        print(client.get('/health').json())
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
