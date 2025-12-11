from fastapi import FastAPI, HTTPException
from app.model import SentimentModel
from app.schemas import PredictRequest, PredictResponse, BatchRequest

app = FastAPI(title="ML Sentiment API")

model = SentimentModel()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return model.predict(req.text)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return [model.predict(t) for t in req.texts]
