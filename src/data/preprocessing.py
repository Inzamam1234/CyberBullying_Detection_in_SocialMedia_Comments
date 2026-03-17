import pandas as pd
import re

def clean_text(text):

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove usernames
    text = re.sub(r"@\w+", "", text)

    # remove emojis and special characters
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

    # slang normalization
    slang_map = {
        "u": "you",
        "ur": "your",
        "r": "are",
        "idk": "i do not know",
        "wtf": "what the fuck"
    }

    for slang, full in slang_map.items():
        text = re.sub(rf"\b{slang}\b", full, text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_dataset():

    df = pd.read_csv("data/raw/train.csv")

    df["clean_comment"] = df["comment_text"].apply(clean_text)

    # 🔍 VERIFY CLEANING HERE
    for i in range(5):
        print("ORIGINAL:", df["comment_text"][i])
        print("CLEANED :", df["clean_comment"][i])
        print()

    df.to_csv("data/processed/cleaned_train.csv", index=False)

    print("Dataset cleaned and saved.")

if __name__ == "__main__":
    preprocess_dataset()