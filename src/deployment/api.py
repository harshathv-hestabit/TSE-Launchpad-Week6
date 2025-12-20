import uuid
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_path: str
    log_path: str
    
    model_config = SettingsConfigDict(
        env_file= "src/deployment/.env",
        extra= "ignore"
    )

settings = Settings()

MODEL_PATH = Path(settings.model_path)
LOG_PATH = Path(settings.log_path)

ORDINAL_CLASSES = {
    0: "Very Poor",
    1: "Poor",
    2: "Average",
    3: "Good",
    4: "Excellent"
}

try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = FastAPI(
    title="Movie Average Rating Prediction API",
    version="1.0.0"
)

GENRE_COLUMNS = [
    "Action", "Adult", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir",
    "Game-Show", "History", "Horror", "Music", "Musical", "Mystery",
    "News", "Reality-TV", "Romance", "Sci-Fi", "Sport", "Talk-Show",
    "Thriller", "Unknown", "War", "Western"
]

FEATURE_COLUMNS = ["runtimeMinutes", "numVotes"] + GENRE_COLUMNS

class PredictionRequest(BaseModel):
    runtimeMinutes: float = Field(..., gt=0)
    numVotes: int = Field(..., ge=0)
    genres: list[str]

def build_feature_vector(payload: PredictionRequest) -> pd.DataFrame:
    data = {c: 0 for c in FEATURE_COLUMNS}
    data["runtimeMinutes"] = payload.runtimeMinutes
    data["numVotes"] = payload.numVotes

    unknown = False
    for g in payload.genres:
        if g in GENRE_COLUMNS:
            data[g] = 1
        else:
            unknown = True

    if unknown:
        data["Unknown"] = 1

    return pd.DataFrame([data])

def log_prediction(req_id, payload, cls, label):
    df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "request_id": req_id,
        "runtimeMinutes": payload.runtimeMinutes,
        "numVotes": payload.numVotes,
        "genres": ",".join(payload.genres),
        "predicted_class": cls,
        "predicted_label": label
    }])
    df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        X = build_feature_vector(payload)
        proba = model_pipeline.predict_proba(X)[0]
        cls = int(proba.argmax())
        label = ORDINAL_CLASSES[cls]

        req_id = str(uuid.uuid4())
        log_prediction(req_id, payload, cls, label)

        return {
            "request_id": req_id,
            "predicted_class": cls,
            "predicted_label": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))