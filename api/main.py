from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.predictor import ToxicityPredictor

app = FastAPI()

# ✅ CORS MUST BE HERE (AFTER app = FastAPI())
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


@app.post("/predict")
def predict(req: CommentRequest):
    return predictor.predict(req.text)

@app.get("/")
def home():
    return {"message": "API running"}