"""
src/predictor.py
Shared prediction class used by BOTH api/main.py and app/streamlit_app.py.
Auto-downloads model from HuggingFace Hub if not found locally.
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_PATH, MODEL_DIR, THRESHOLD_PATH,
    BERT_MODEL_NAME, MAX_LEN, LABEL_COLS, THRESHOLDS, HF_REPO_ID
)


# ── Model definition (single place in entire project) ─────────
class ToxicCommentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


# ── Auto-download weights if not present ──────────────────────
def ensure_model_files():
    """
    Download model weights and thresholds from Hugging Face Hub
    if they don't exist locally. This runs automatically on
    first startup in deployed environments.
    """
    model_missing     = not MODEL_PATH.exists()
    threshold_missing = not THRESHOLD_PATH.exists()

    if not (model_missing or threshold_missing):
        return  # Everything already present locally

    if not HF_REPO_ID:
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH} and HF_REPO_ID is not set.\n"
            "Either place toxic_model.pt in models/checkpoints/ "
            "or set the HF_REPO_ID environment variable."
        )

    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if model_missing:
        print(f"[Predictor] Downloading model weights from {HF_REPO_ID}...")
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="toxic_model.pt",
            local_dir=str(MODEL_DIR),
        )
        print(f"[Predictor] Weights saved to {path}")

    if threshold_missing:
        print(f"[Predictor] Downloading thresholds from {HF_REPO_ID}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="thresholds.json",
            local_dir=str(MODEL_DIR),
        )
        print("[Predictor] Thresholds saved.")


# ── Predictor class ───────────────────────────────────────────
class ToxicityPredictor:
    """
    Load once, call predict() or predict_batch() many times.
    """

    def __init__(self):
        # Auto-download weights if running in cloud
        ensure_model_files()

        # Reload thresholds (in case they were just downloaded)
        import json
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = THRESHOLDS  # fallback from config

        self.device = torch.device(
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        print(f"[Predictor] Using device: {self.device}")

        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

        self.model = ToxicCommentClassifier()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print("[Predictor] Model loaded successfully ✅")

    # ── Internal helper ───────────────────────────────────────
    def _probs_to_result(self, probs_row) -> dict:
        scores    = {l: round(float(p), 4) for l, p in zip(LABEL_COLS, probs_row)}
        flags     = {l: bool(scores[l] >= self.thresholds[l]) for l in LABEL_COLS}
        triggered = [l for l, flag in flags.items() if flag]
        return {
            "scores":   scores,
            "flags":    flags,
            "is_toxic": len(triggered) > 0,
            "labels":   triggered,
        }

    # ── Single prediction ─────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Returns:
        {
            "scores":   {"toxic": 0.92, ...},
            "flags":    {"toxic": True, ...},
            "is_toxic": True,
            "labels":   ["toxic", "insult"]
        }
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=MAX_LEN,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.sigmoid(self.model(**inputs)).cpu().numpy()[0]

        return self._probs_to_result(probs)

    # ── Batch prediction ──────────────────────────────────────
    def predict_batch(self, texts: list) -> list:
        """
        Predict multiple texts in one forward pass.
        Much faster than calling predict() in a loop.
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            truncation=True, padding=True, max_length=MAX_LEN,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.sigmoid(self.model(**inputs)).cpu().numpy()

        return [self._probs_to_result(row) for row in probs]


# ── Quick smoke test ──────────────────────────────────────────
if __name__ == "__main__":
    predictor = ToxicityPredictor()

    tests = [
        "You are stupid and ugly",
        "This is a really nice photo!",
        "I will find you and hurt you",
        "Great content, keep it up!",
        "Go kill yourself you worthless idiot",
    ]

    print("\n--- Smoke Test ---")
    for text in tests:
        r = predictor.predict(text)
        status = "🚨 TOXIC" if r["is_toxic"] else "✅ CLEAN"
        labels = f"  → {', '.join(r['labels'])}" if r["labels"] else ""
        print(f"{status}  {text}{labels}")

    print("\n--- Batch Test ---")
    results = predictor.predict_batch(tests)
    toxic_count = sum(1 for r in results if r["is_toxic"])
    print(f"Processed {len(tests)} texts, found {toxic_count} toxic.")