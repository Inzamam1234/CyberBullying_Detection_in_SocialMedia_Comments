"""
Threshold Fixer
Finds thresholds that give balanced Precision/Recall (F1 with beta=1)
and also shows you the full precision-recall tradeoff curve
Run this ONCE, it overwrites thresholds.json
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import json

# ── Device ───────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

# ── Model ─────────────────────────────────────────────────────
class ToxicCommentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

# ── Dataset ───────────────────────────────────────────────────
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/cleaned_train.csv")
texts = df["clean_comment"].astype(str).tolist()
labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(
    texts, padding=True, truncation=True,
    max_length=128, return_tensors="pt"
)

dataset = ToxicDataset(encodings, labels)
torch.manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=32)

# ── Load model ────────────────────────────────────────────────
print("Loading model...")
model = ToxicCommentClassifier()
model.load_state_dict(
    torch.load("models/checkpoints/toxic_model1.pt", map_location=device)
)
model.to(device)
model.eval()

# ── Get predictions ───────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask)
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

all_preds  = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# ── Find balanced thresholds ──────────────────────────────────
print("\n" + "="*60)
print("Finding balanced thresholds (precision ≈ recall)...")
print("="*60)

best_thresholds = {}

for i, label in enumerate(LABEL_COLS):
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
        all_labels[:, i], all_preds[:, i]
    )

    # F1 at every threshold point
    f1_scores = np.where(
        (precision_arr + recall_arr) == 0, 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr)
    )

    # Best balanced threshold
    best_idx    = np.argmax(f1_scores)
    best_thresh = float(thresholds_arr[best_idx]) if best_idx < len(thresholds_arr) else 0.5
    best_thresh = round(best_thresh, 3)

    preds_binary = (all_preds[:, i] > best_thresh).astype(int)
    p = (preds_binary * all_labels[:, i]).sum() / (preds_binary.sum() + 1e-9)
    r = (preds_binary * all_labels[:, i]).sum() / (all_labels[:, i].sum() + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)

    best_thresholds[label] = best_thresh

    pos_count = int(all_labels[:, i].sum())
    print(f"\n  {label}")
    print(f"    Positive samples : {pos_count}")
    print(f"    Best threshold   : {best_thresh:.3f}")
    print(f"    Precision        : {p:.3f}")
    print(f"    Recall           : {r:.3f}")
    print(f"    F1               : {f1:.3f}")

# ── Save ──────────────────────────────────────────────────────
with open("models/checkpoints/thresholds.json", "w") as f:
    json.dump(best_thresholds, f, indent=2)

print("\n" + "="*60)
print("✅ Balanced thresholds saved to models/checkpoints/thresholds.json")
print("="*60)
print("\nFinal thresholds:")
for label, thresh in best_thresholds.items():
    print(f"  {label:<20} {thresh}")