"""
upload_model.py
Run this ONCE to upload your trained weights to Hugging Face Hub.
After this, your deployed API will download weights automatically on startup.

Steps before running:
1. Create account at huggingface.co
2. Go to huggingface.co/settings/tokens → create a token with WRITE access
3. Run: huggingface-cli login   (paste your token)
4. Run: python upload_model.py
"""

from huggingface_hub import HfApi, create_repo
import os

# ── CHANGE THESE TWO LINES ────────────────────────────────────
HF_USERNAME = "Injamam11"   # e.g. "inzamam"
REPO_NAME    = "cyberbullying-detector"
# ─────────────────────────────────────────────────────────────

REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

# Create the repo (private=False means public — fine for a project)
print(f"Creating repo: {REPO_ID}")
create_repo(REPO_ID, exist_ok=True, repo_type="model")

# Upload model weights
print("Uploading toxic_model1.pt ...")
api.upload_file(
    path_or_fileobj="models/checkpoints/toxic_model1.pt",
    path_in_repo="toxic_model1.pt",
    repo_id=REPO_ID,
    repo_type="model",
)

# Upload thresholds
print("Uploading thresholds.json ...")
api.upload_file(
    path_or_fileobj="models/checkpoints/thresholds.json",
    path_in_repo="thresholds.json",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n✅ Model uploaded successfully!")
print(f"   View at: https://huggingface.co/Injamam11")
print(f"\n   Set this in your .env file:")
print(f"   HF_REPO_ID={REPO_ID}")