"""
app/streamlit_app.py — Premium UI Version
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import ToxicityPredictor
from config import LABEL_COLS


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered",
)

# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_predictor():
    return ToxicityPredictor()

predictor = load_predictor()


# ===============================
# PREMIUM CSS
# ===============================

st.markdown("""
<style>

/* Cards */
.card {
    padding: 14px;
    border-radius: 12px;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    color: #111 !important;   /* ✅ FIX TEXT COLOR */
}

/* Toxic */
.toxic {
    background: #ffe5e5;
    border-left: 5px solid #ff4d4f;
    color: #111 !important;   /* ✅ FORCE DARK TEXT */
}

/* Safe */
.safe {
    background: #e6f7ec;
    border-left: 5px solid #2ecc71;
    color: #111 !important;   /* ✅ FORCE DARK TEXT */
}

/* Chips */
.chip {
    display:inline-block;
    background:#ff4d4f;
    color:white;
    padding:4px 10px;
    border-radius:20px;
    font-size:0.75rem;
    margin:3px;
}

/* Blur */
.blur {
    filter: blur(5px);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #2ecc71, #ff4d4f);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# HEADER
# ===============================
st.markdown("""
<style>

/* Hero container */
.hero {
    text-align: center;
    padding: 30px 20px 10px 20px;
}

/* Title */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 8px;
    color: #ffffff;
}

/* Subtitle */
.hero-subtitle {
    font-size: 1.1rem;
    color: #9aa0a6;
    margin-top: 0;
}

/* Divider spacing */
.hero-divider {
    margin-top: 20px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-title">
        🛡️ Cyberbullying Detection System
    </div>
    <div class="hero-subtitle">
        AI-powered toxic comment detection using DistilBERT
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
st.divider()

# ===============================
# TABS
# ===============================

tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📋 Feed Demo", "ℹ️ About"])


# ============================================================
# TAB 1 — SINGLE ANALYSIS
# ============================================================

with tab1:

    user_input = st.text_area(
        "Enter a comment:",
        placeholder="Type something...",
        height=120
    )

    analyze = st.button("Analyze", type="primary")

    if analyze:
        if not user_input.strip():
            st.warning("Please enter a comment")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict(user_input)

            # RESULT
            if result["is_toxic"]:
                st.markdown(
                    '<div class="card toxic">'
                    '<b>⚠️ Toxic Comment Detected</b><br>' +
                    "".join(f'<span class="chip">{l}</span>' for l in result["labels"]) +
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="card safe"><b>✅ Safe Comment</b></div>',
                    unsafe_allow_html=True
                )

            # SCORES
            st.subheader("📊 Score Breakdown")

            for label in LABEL_COLS:
                score = result["scores"][label]
                flagged = result["flags"][label]

                col1, col2, col3 = st.columns([3, 6, 1])

                with col1:
                    icon = "🔴" if flagged else "🟢"
                    name = label.replace("_", " ").title()
                    st.write(f"{icon} {name}")

                with col2:
                    st.progress(score)

                with col3:
                    st.write(f"{score:.2f}")


# ============================================================
# TAB 2 — FEED DEMO
# ============================================================

with tab2:

    st.subheader("Simulated Social Media Feed")

    default_comments = [
        "Nice post, I really enjoyed reading this!",
        "You are stupid and ugly",
        "Great content, keep it up 👍",
        "I will find you and hurt you",
        "This was really helpful, thank you!",
        "Go kill yourself, you worthless",
        "Interesting perspective, I learned something new",
        "You're such an idiot",
    ]

    user_comments = st.text_area(
        "Add your own comments (one per line):",
        height=100
    )

    comments = default_comments.copy()

    if user_comments.strip():
        extra = [c.strip() for c in user_comments.split("\n") if c.strip()]
        comments = extra + comments

    if st.button("Run Analysis", type="primary"):

        with st.spinner("Analyzing feed..."):
            results = predictor.predict_batch(comments)

        toxic_count = sum(r["is_toxic"] for r in results)
        safe_count = len(results) - toxic_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("Toxic", toxic_count)
        c3.metric("Safe", safe_count)

        st.divider()

        for text, res in zip(comments, results):

            if res["is_toxic"]:
                st.markdown(
                    '<div class="card toxic">'
                    f'<b>🚨 Hidden:</b> <span class="blur">{text}</span><br>' +
                    "".join(f'<span class="chip">{l}</span>' for l in res["labels"]) +
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="card safe">💬 {text}</div>',
                    unsafe_allow_html=True
                )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab3:

    st.subheader("About")

    st.markdown("""
    **Cyberbullying Detection System**

    Detects toxic comments using a fine-tuned DistilBERT model.

    **Model**
    - DistilBERT base
    - Multi-label classification
    - ROC-AUC ≈ 0.99

    **Labels**
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate

    **Tech Stack**
    - PyTorch
    - HuggingFace
    - FastAPI
    - Streamlit
    """)
