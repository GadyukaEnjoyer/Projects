from __future__ import annotations

import pickle
from contextlib import asynccontextmanager

from clearml import Model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Вставить сюда model_id после register_model.py ---
REGISTRY_MODEL_ID = "00dd79e58b6e420385501ada6fd9ec18"


class InferenceBundle:
    def __init__(self) -> None:
        self.classifier = None
        self.vectorizer = None

    def load_from_registry(self, model_id: str) -> None:
        weights_path = Model(model_id=model_id).get_local_copy()
        with open(weights_path, "rb") as handle:
            payload = pickle.load(handle)
        self.classifier = payload["model"]
        self.vectorizer = payload["vectorizer"]


bundle = InferenceBundle()


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Загрузка весов из ClearML Registry...")
    bundle.load_from_registry(REGISTRY_MODEL_ID)
    print("Inference-модель готова.")
    yield


app = FastAPI(title="Social Media Sentiment API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст поста из соцсети")


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
def root() -> dict[str, str | list[str]]:
    return {
        "service": "Social Media Sentiment API",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "model_source": "clearml-registry"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if bundle.classifier is None or bundle.vectorizer is None:
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")

    vectorized = bundle.vectorizer.transform([payload.text.strip()])
    label = bundle.classifier.predict(vectorized)[0]
    confidence = float(bundle.classifier.predict_proba(vectorized).max())

    return PredictResponse(label=label, confidence=round(confidence, 3))
