from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/").status_code == 200

def test_predict():
    res = client.post("/predict", json={"text": "I love this!"})
    assert res.status_code == 200
