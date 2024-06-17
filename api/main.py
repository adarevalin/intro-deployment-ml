from fastapi import FastAPI
from .app.models import PredictionResponse, PredictionRequest
from .app.views import get_prediction

app = FastAPI()

@app.post('/v1/prediction', response_model=PredictionResponse)

def make_model_prediction(request: PredictionRequest):
    predicted_value = get_prediction(request)
    return PredictionResponse(worldwide_gross=predicted_value)