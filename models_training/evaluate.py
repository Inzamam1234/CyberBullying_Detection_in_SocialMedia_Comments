"""
Evaluation Script — Run this after training
Gives you: ROC-AUC, F1, Precision, Recall per label
+ finds the best threshold per label
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report
)
import numpy as np
import json

# ── Device ──────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ── Label columns ────────────────────────────────────────────
LABEL_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

# ── Model definition ─────────────────────────────────────────
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

# ── Dataset ──────────────────────────────────────────────────
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
print("\nLoading data...")
df = pd.read_csv("data/processed/cleaned_train.csv")
texts = df["clean_comment"].astype(str).tolist()
labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(
    texts, padding=True, truncation=True,
    max_length=128, return_tensors="pt"
)

dataset = ToxicDataset(encodings, labels)

# Use the SAME split as training (same random seed = same val set)
torch.manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=32)

# ── Load trained model ────────────────────────────────────────
print("Loading model...")
model = ToxicCommentClassifier()
model.load_state_dict(
    torch.load("models/checkpoints/toxic_model1.pt", map_location=device)
)
model.to(device)
model.eval()

# ── Run predictions ───────────────────────────────────────────
print("Running predictions on validation set...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(true_labels)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# ── ROC-AUC ───────────────────────────────────────────────────
roc_auc = roc_auc_score(all_labels, all_preds, average="macro")
print(f"\n{'='*50}")
print(f"  Overall ROC-AUC (macro): {roc_auc:.4f}")
print(f"{'='*50}")

# Per-label ROC-AUC
print("\nPer-label ROC-AUC:")
for i, label in enumerate(LABEL_COLS):
    try:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        print(f"  {label:<20} {auc:.4f}")
    except Exception:
        print(f"  {label:<20} N/A (no positive samples)")

# ── Find best threshold per label ─────────────────────────────
print("\nFinding best threshold per label (maximizing F1)...")
best_thresholds = {}

for i, label in enumerate(LABEL_COLS):
    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in np.arange(0.10, 0.90, 0.05):
        preds_binary = (all_preds[:, i] > thresh).astype(int)
        f1 = f1_score(all_labels[:, i], preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    best_thresholds[label] = round(float(best_thresh), 2)
    print(f"  {label:<20} threshold={best_thresholds[label]:.2f}  F1={best_f1:.4f}")

# ── Save thresholds ────────────────────────────────────────────
with open("models/checkpoints/thresholds.json", "w") as f:
    json.dump(best_thresholds, f, indent=2)
print("\n✅ Thresholds saved to models/checkpoints/thresholds.json")

# ── Full classification report at best thresholds ─────────────
print(f"\n{'='*50}")
print("Classification Report (at best thresholds):")
print(f"{'='*50}")

for i, label in enumerate(LABEL_COLS):
    thresh = best_thresholds[label]
    preds_binary = (all_preds[:, i] > thresh).astype(int)
    p = precision_score(all_labels[:, i], preds_binary, zero_division=0)
    r = recall_score(all_labels[:, i], preds_binary, zero_division=0)
    f1 = f1_score(all_labels[:, i], preds_binary, zero_division=0)
    print(f"  {label:<20} Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

print(f"\n{'='*50}")
print("Evaluation complete!")
print(f"{'='*50}")