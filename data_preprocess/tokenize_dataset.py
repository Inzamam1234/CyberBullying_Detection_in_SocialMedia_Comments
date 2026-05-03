# ---------------------------------------------
# Import required libraries
# ---------------------------------------------
import pandas as pd
import torch
from transformers import DistilBertTokenizer


# ---------------------------------------------
# Step 1: Load the cleaned dataset
# ---------------------------------------------
# This file was created during preprocessing
df = pd.read_csv("data/processed/cleaned_train.csv")

# Display first 5 rows to verify the dataset
print("Dataset preview:")
print(df.head())


# ---------------------------------------------
# Step 2: Load the DistilBERT tokenizer
# ---------------------------------------------
# This tokenizer converts text into tokens
# that the DistilBERT model can understand
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# ---------------------------------------------
# Step 3: Extract text data
# ---------------------------------------------
# Convert the clean comments column into a list
texts = df["clean_comment"].astype(str).tolist()


# ---------------------------------------------
# Step 4: Define label columns
# ---------------------------------------------
# These are the target classes for the model
label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Convert labels into a NumPy array
labels = df[label_cols].values


# ---------------------------------------------
# Step 5: Define maximum sequence length
# ---------------------------------------------
# DistilBERT typically works well with 128 tokens
MAX_LEN = 128


# ---------------------------------------------
# Step 6: Tokenize the text
# ---------------------------------------------
# This converts text into:
# - input_ids
# - attention_mask
encodings = tokenizer(
    texts,
    padding=True,        # pad shorter sentences
    truncation=True,     # cut longer sentences
    max_length=MAX_LEN,
    return_tensors="pt"  # return PyTorch tensors
)


# ---------------------------------------------
# Step 7: Inspect tokenization results
# ---------------------------------------------
print("\nEncoding keys:")
print(encodings.keys())

print("\nInput IDs shape:")
print(encodings["input_ids"].shape)

print("\nAttention mask shape:")
print(encodings["attention_mask"].shape)


# ---------------------------------------------
# Step 8: Show tokenization example
# ---------------------------------------------
sample_text = texts[0]

tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nExample text:")
print(sample_text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)


# ---------------------------------------------
# Step 9: Save processed tensors
# ---------------------------------------------
# Save tokenized inputs for training
torch.save(encodings, "data/processed/encodings.pt")

# Save labels for training
torch.save(labels, "data/processed/labels.pt")

print("\nTokenized data saved successfully!")