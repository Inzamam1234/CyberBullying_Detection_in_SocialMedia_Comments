"""
api/main.py — Production FastAPI backend
Run with: uvicorn api.main:app --reload
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import ToxicityPredictor

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Startup: load model once ──────────────────────────────────
predictor: ToxicityPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    predictor = ToxicityPredictor()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Cyberbullying Detection API",
    description="Multi-label toxicity detection using DistilBERT",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
# In production, replace * with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this after deployment
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Request / Response schemas ────────────────────────────────
class TextInput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty")
        if len(v) > 2000:
            raise ValueError("text too long (max 2000 characters)")
        return v


class BatchInput(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > 50:
            raise ValueError("max 50 texts per batch")
        return [t.strip() for t in v if t.strip()]


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Cyberbullying Detection API is running"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(data: TextInput, request: Request):
    """
    Predict toxicity for a single comment.

    Returns:
    - scores: probability per label (0-1)
    - flags: True/False per label (based on tuned thresholds)
    - is_toxic: True if ANY label is flagged
    - labels: list of triggered label names
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()
    try:
        result = predictor.predict(data.text)
        elapsed = round(time.time() - start, 3)
        logger.info(
            f"predict | is_toxic={result['is_toxic']} "
            f"| labels={result['labels']} | {elapsed}s"
        )
        return {**result, "elapsed_seconds": elapsed}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_batch")
async def predict_batch(data: BatchInput):
    """
    Predict toxicity for multiple comments in one call.
    More efficient than calling /predict in a loop.
    Max 50 texts per request.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()
    try:
        results = predictor.predict_batch(data.texts)
        elapsed = round(time.time() - start, 3)
        logger.info(f"predict_batch | n={len(data.texts)} | {elapsed}s")
        return {"results": results, "elapsed_seconds": elapsed}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")