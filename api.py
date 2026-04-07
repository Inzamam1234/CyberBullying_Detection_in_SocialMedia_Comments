from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

app = FastAPI()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model
class ToxicCommentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

# Load model
model = ToxicCommentClassifier()
model.load_state_dict(torch.load("models/checkpoints/toxic_model.pt", map_location=device))
model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    return {label: float(prob) for label, prob in zip(labels, probs)}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)