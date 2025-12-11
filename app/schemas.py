from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    processing_time_ms: float

class BatchRequest(BaseModel):
    texts: List[str]
