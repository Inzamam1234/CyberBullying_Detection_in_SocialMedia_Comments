"""
Evaluation & Prediction Script
"""

# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import roc_auc_score
import numpy as np

# ===============================
# 2. DEVICE
# ===============================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ===============================
# 3. MODEL DEFINITION (SAME AS BEFORE)
# ===============================

class ToxicCommentClassifier(nn.Module):

    def __init__(self):
        super(ToxicCommentClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 6)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]

        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits


# ===============================
# 4. LOAD DATA
# ===============================

df = pd.read_csv("data/processed/cleaned_train.csv")

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

texts = df["clean_comment"].astype(str).tolist()
labels = df[label_cols].values

labels = torch.tensor(labels, dtype=torch.float)


# ===============================
# 5. TOKENIZATION
# ===============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)


# ===============================
# 6. DATASET CLASS
# ===============================

class ToxicDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


dataset = ToxicDataset(encodings, labels)


# ===============================
# 7. SPLIT DATA (SAME AS TRAINING)
# ===============================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(dataset, [train_size, val_size])


val_loader = DataLoader(val_dataset, batch_size=32)


# ===============================
# 8. LOAD TRAINED MODEL
# ===============================

model = ToxicCommentClassifier()
model.load_state_dict(torch.load("models/checkpoints/toxic_model.pt", map_location=device))
model.to(device)

model.eval()

print("Model loaded successfully")


# ===============================
# 9. PREDICTIONS
# ===============================

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].cpu().numpy()

        outputs = model(input_ids, attention_mask)

        probs = torch.sigmoid(outputs).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(labels_batch)


all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)


# ===============================
# 10. ROC-AUC SCORE
# ===============================

roc_auc = roc_auc_score(all_labels, all_preds, average="macro")

print("\n🔥 ROC-AUC Score:", roc_auc)


# ===============================
# 11. TEST CUSTOM COMMENT
# ===============================

def predict_comment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    result = dict(zip(label_cols, probs))

    return result


# ===============================
# 12. TEST EXAMPLES
# ===============================

test_text = "You are stupid and ugly"

result = predict_comment(test_text)

print("\nTest Comment:", test_text)
print("Prediction:", result)