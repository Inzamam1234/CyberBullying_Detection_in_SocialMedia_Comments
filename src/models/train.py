"""
Training Pipeline for Cyberbullying Detection Model

This script:
1. Loads cleaned dataset
2. Tokenizes text using DistilBERT tokenizer
3. Builds Dataset & DataLoader
4. Loads DistilBERT classification model
5. Trains the model
6. Saves trained model
"""

# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer
from transformers import DistilBertTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from model import ToxicCommentClassifier


# ===============================
# 2. DEVICE SETUP (Mac M-series)
# ===============================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ===============================
# 3. LOAD DATASET
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
# 4. TOKENIZATION
# ===============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

MAX_LEN = 128

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)


# ===============================
# 5. DATASET CLASS
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
# 6. TRAIN / VALIDATION SPLIT
# ===============================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# ===============================
# 7. DATALOADERS
# ===============================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


# ===============================
# 8. LOAD MODEL
# ===============================

model = ToxicCommentClassifier()

model.to(device)


# ===============================
# 9. LOSS FUNCTION
# ===============================

criterion = nn.BCEWithLogitsLoss()


# ===============================
# 10. OPTIMIZER
# ===============================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)


# ===============================
# 11. TRAINING LOOP
# ===============================

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f"Epoch {epoch+1} Loss {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Training Loss: {avg_loss:.4f}")


# ===============================
# 12. SAVE MODEL
# ===============================

torch.save(
    model.state_dict(),
    "models/checkpoints/toxic_model.pt"
)

print("\nModel saved successfully!")