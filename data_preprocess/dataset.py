"""
Dataset & DataLoader Pipeline for Cyberbullying Detection

This script:
1. Loads the cleaned dataset
2. Tokenizes text using DistilBERT tokenizer
3. Converts labels to tensors
4. Creates a custom PyTorch Dataset
5. Splits dataset into train and validation sets
6. Creates DataLoaders for model training
"""

# ================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer


# ================================
# 2. LOAD CLEANED DATASET
# ================================

# Load the dataset produced in Phase 3
df = pd.read_csv("data/processed/cleaned_train.csv")

print("Dataset Loaded")
print(df.head())


# ================================
# 3. DEFINE LABEL COLUMNS
# ================================

# These are the toxicity categories
label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ================================
# 4. EXTRACT TEXT AND LABELS
# ================================

# Convert text column to list
texts = df["clean_comment"].astype(str).tolist()

# Extract labels as numpy array
labels = df[label_cols].values

# Convert labels to PyTorch tensor
labels = torch.tensor(labels, dtype=torch.float)


# ================================
# 5. LOAD DISTILBERT TOKENIZER
# ================================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Maximum token length for comments
MAX_LEN = 128


# ================================
# 6. TOKENIZE TEXT
# ================================

# Convert text to token IDs and attention masks
encodings = tokenizer(
    texts,
    padding=True,          # pad shorter sentences
    truncation=True,       # truncate long sentences
    max_length=MAX_LEN,    # maximum sequence length
    return_tensors="pt"    # return PyTorch tensors
)

print("\nTokenization Complete")
print(encodings.keys())
print(encodings["input_ids"].shape)
print(encodings["attention_mask"].shape)


# ================================
# 7. CREATE CUSTOM DATASET CLASS
# ================================

class ToxicCommentDataset(Dataset):
    """
    Custom Dataset class for PyTorch
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """
        Returns number of samples
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns one training example
        """

        # Extract tokenized input
        item = {key: val[idx] for key, val in self.encodings.items()}

        # Add label
        item["labels"] = self.labels[idx]

        return item


# ================================
# 8. CREATE DATASET OBJECT
# ================================

dataset = ToxicCommentDataset(encodings, labels)

print("\nDataset Size:", len(dataset))


# ================================
# 9. TRAIN / VALIDATION SPLIT
# ================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("\nTrain Size:", train_size)
print("Validation Size:", val_size)


# ================================
# 10. CREATE DATALOADERS
# ================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoaders Created")


# ================================
# 11. TEST BATCH OUTPUT
# ================================

batch = next(iter(train_loader))

print("\nBatch Shapes:")
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("labels:", batch["labels"].shape)


# ================================
# 12. DISPLAY SAMPLE LABEL
# ================================

print("\nExample Label Vector:")
print(batch["labels"][0])

print("\nDataset Pipeline Ready for Training")