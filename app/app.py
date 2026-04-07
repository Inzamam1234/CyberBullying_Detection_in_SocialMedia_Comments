import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


# ===============================
# 1. DEVICE
# ===============================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===============================
# 2. MODEL DEFINITION
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
# 3. LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    model = ToxicCommentClassifier()
    model.load_state_dict(torch.load("models/checkpoints/toxic_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


# ===============================
# 4. TOKENIZER
# ===============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# ===============================
# 5. LABELS
# ===============================

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ===============================
# 6. PREDICTION FUNCTION
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

    # convert to normal float
    result = {label: float(prob) for label, prob in zip(label_cols, probs)}

    return result


# ===============================
# 7. SIMPLE REWRITE SUGGESTION
# ===============================

def suggest_rewrite(text):
    return "Try expressing your opinion politely 😊"


# ===============================
# 8. STREAMLIT UI
# ===============================

st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

st.title("🛡️ Cyberbullying Detection System")
st.write("Detect toxic comments and suggest safer alternatives")

# Input box
user_input = st.text_area("💬 Enter your comment:")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment")
    else:
        result = predict_comment(user_input)

        st.subheader("📊 Prediction Scores")

        # Show scores
        for label, score in result.items():
            st.write(f"{label}: {score:.2f}")

        # Detect toxicity
        if result["toxic"] > 0.5 or result["insult"] > 0.5:

            st.error("⚠️ Toxic Comment Detected!")

            st.subheader("💡 Suggested Rewrite")
            st.info(suggest_rewrite(user_input))

        else:
            st.success("✅ This comment looks safe!")

st.subheader("🌐 Simulated Social Media Feed")

comments = [
    "Nice post, I like it!",
    "You are stupid and ugly",
    "Great content bro!",
    "Nobody likes you",
]

for comment in comments:

    result = predict_comment(comment)

    if result["toxic"] > 0.5 or result["insult"] > 0.5:
        st.markdown(f"❌ **[Hidden Toxic Comment]**")
    else:
        st.markdown(f"💬 {comment}")