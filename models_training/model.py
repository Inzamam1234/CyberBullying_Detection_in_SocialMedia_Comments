"""
DistilBERT Model for Cyberbullying Detection

This module defines the neural network architecture used for
multi-label classification of toxic comments.

Labels predicted:
1. toxic
2. severe_toxic
3. obscene
4. threat
5. insult
6. identity_hate
"""

# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import torch
import torch.nn as nn
from transformers import DistilBertModel


# ===============================
# 2. MODEL CLASS
# ===============================

class ToxicCommentClassifier(nn.Module):

    def __init__(self):
        """
        Initialize the model layers
        """

        super(ToxicCommentClassifier, self).__init__()

        # Load pretrained DistilBERT model
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Hidden embedding size of DistilBERT
        hidden_size = self.bert.config.hidden_size   # 768

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

        # Final classification layer
        # Output = 6 toxicity categories
        self.classifier = nn.Linear(hidden_size, 6)


    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model

        Parameters:
        input_ids      -> tokenized comment
        attention_mask -> padding mask

        Returns:
        logits -> raw prediction scores
        """

        # Pass inputs through DistilBERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract contextual embeddings
        hidden_state = outputs.last_hidden_state

        # Use representation of first token ([CLS])
        pooled_output = hidden_state[:, 0]

        # Apply dropout
        x = self.dropout(pooled_output)

        # Final classification layer
        logits = self.classifier(x)

        return logits


# ===============================
# 3. TEST MODEL (OPTIONAL)
# ===============================

if __name__ == "__main__":

    # Create model instance
    model = ToxicCommentClassifier()

    # Print model architecture
    print(model)

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones((2, 128))

    # Forward pass
    outputs = model(input_ids, attention_mask)

    print("\nOutput shape:", outputs.shape)