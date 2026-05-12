from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.predictor import ToxicityPredictor

app = FastAPI()

# CORS (VERY IMPORTANT for extension)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ToxicityPredictor()


class CommentRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "API running ✅"}


@app.post("/predict")
def predict(req: CommentRequest):
    result = predictor.predict(req.text)

    # IMPORTANT: simplify response for extension
    return {
        "is_toxic": result["is_toxic"],
        "labels": result["labels"],
        "scores": result["scores"]
    }