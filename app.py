"""
app.py
------
This is the Streamlit web application for Fake News Detection.

WHAT IT DOES (in plain English):
1. Loads the model and TF-IDF vectorizer that train.py already trained and saved.
2. Shows a text box where the user can paste a news headline or article.
3. When the user clicks "Predict", it:
   - cleans the text the same way train.py did
   - converts it to TF-IDF numbers using the SAME vectorizer used in training
   - asks the trained Logistic Regression model for a prediction
   - shows FAKE or REAL, along with a confidence percentage

Run this file with:
    streamlit run app.py
"""

import os
import re
import string

import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Load stopwords (same logic as train.py, so cleaning matches exactly)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            try:
                nltk.download("stopwords")
            except Exception:
                pass
        return set(stopwords.words("english"))
    except Exception:
        return {
            "the", "a", "an", "and", "or", "but", "if", "then", "is", "are",
            "was", "were", "be", "been", "being", "to", "of", "in", "on",
            "for", "with", "as", "by", "at", "from", "this", "that", "it",
            "its", "he", "she", "they", "them", "his", "her", "their", "we",
            "you", "your", "i", "not", "so", "no", "do", "does", "did",
        }


STOPWORDS = load_stopwords()


def clean_text(text):
    """Cleans input text exactly the same way it was cleaned during training."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Load the trained model + vectorizer (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()

# ---------------------------------------------------------------------------
# Simple custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1.8rem;
    }
    .result-fake {
        background-color: #ffe3e3;
        border: 1px solid #ff8787;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-real {
        background-color: #e6fcf5;
        border: 1px solid #63e6be;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Paste a news headline or article below and let the '
    'Machine Learning model predict if it is REAL or FAKE.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Two tabs:
#   1) Predict     -> the ORIGINAL functionality, unchanged
#   2) Model Comparison -> NEW: shows results from train_all_models.py
#                           (all supervised models, ANN, and clustering)
# ---------------------------------------------------------------------------
tab_predict, tab_comparison = st.tabs(["🔍 Predict", "📊 Model Comparison"])

# =============================================================================
# TAB 1: PREDICT (existing functionality - logic unchanged from before)
# =============================================================================
with tab_predict:
    if model is None or vectorizer is None:
        st.error(
            "⚠️ Model files not found in the `model/` folder.\n\n"
            "Please train the model first by running:\n\n"
            "`python train.py`\n\n"
            "Then restart this app."
        )
    else:
        news_text = st.text_area(
            "Enter news headline or article text:",
            height=220,
            placeholder="Paste the news headline or full article text here...",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            predict_clicked = st.button("🔍 Predict", use_container_width=True)
        with col2:
            clear_clicked = st.button("🗑️ Clear", use_container_width=True)

        if clear_clicked:
            st.rerun()

        # ---------------------------------------------------------------
        # Prediction logic
        # ---------------------------------------------------------------
        if predict_clicked:
            if not news_text or not news_text.strip():
                st.warning("Please enter some news text before predicting.")
            else:
                cleaned = clean_text(news_text)

                if cleaned.strip() == "":
                    st.warning(
                        "The text you entered doesn't contain enough meaningful words "
                        "to make a prediction. Please try a longer piece of text."
                    )
                else:
                    # Convert cleaned text to TF-IDF features
                    features = vectorizer.transform([cleaned])

                    # Get prediction (0 = REAL, 1 = FAKE) and probability for each class
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]

                    fake_prob = probabilities[1] * 100
                    real_prob = probabilities[0] * 100

                    st.markdown("### Result")

                    if prediction == 1:
                        confidence = fake_prob
                        st.markdown(
                            f"""
                            <div class="result-fake">
                                <h2>🚨 FAKE NEWS</h2>
                                <h4>Confidence: {confidence:.2f}%</h4>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info(
                            "🧠 **What this means:** Based on the words and patterns in this "
                            "text, the model believes it resembles articles labeled as FAKE "
                            "in its training data (for example, sensational language, "
                            "unverified claims, or emotionally charged wording)."
                        )
                    else:
                        confidence = real_prob
                        st.markdown(
                            f"""
                            <div class="result-real">
                                <h2>✅ REAL NEWS</h2>
                                <h4>Confidence: {confidence:.2f}%</h4>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info(
                            "🧠 **What this means:** Based on the words and patterns in this "
                            "text, the model believes it resembles articles labeled as REAL "
                            "in its training data (for example, neutral tone and "
                            "attributed, fact-style reporting)."
                        )

                    with st.expander("See probability breakdown"):
                        st.write(f"**REAL probability:** {real_prob:.2f}%")
                        st.progress(int(real_prob))
                        st.write(f"**FAKE probability:** {fake_prob:.2f}%")
                        st.progress(int(fake_prob))

                    with st.expander("See cleaned text used by the model"):
                        st.write(cleaned)

    st.markdown("---")
    st.caption(
        "⚠️ Disclaimer: This is a beginner-level educational ML project. "
        "Predictions are based on statistical text patterns learned from a "
        "training dataset and should NOT be treated as fact-checking or a "
        "substitute for verified journalism."
    )

# =============================================================================
# TAB 2: MODEL COMPARISON (NEW - reads results/ produced by train_all_models.py)
# =============================================================================
with tab_comparison:
    RESULTS_DIR = "results"
    supervised_csv = os.path.join(RESULTS_DIR, "final_model_comparison.csv")
    clustering_csv = os.path.join(RESULTS_DIR, "clustering_comparison.csv")
    reports_txt = os.path.join(RESULTS_DIR, "classification_reports.txt")
    chart_png = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    ann_curves_png = os.path.join(RESULTS_DIR, "ann_training_curves.png")

    if not os.path.exists(supervised_csv):
        st.warning(
            "No comparison results found yet.\n\n"
            "Run the following command once from the project folder, then "
            "reload this page:\n\n"
            "`python train_all_models.py`\n\n"
            "This trains Logistic Regression, Naive Bayes, KNN, Decision "
            "Tree, Random Forest, SVM, an ANN, and 3 clustering algorithms, "
            "then saves their results here for this tab to display."
        )
    else:
        st.markdown("### Supervised Models + ANN")
        st.caption(
            "Each model was trained on the same TF-IDF text features and "
            "evaluated on the same held-out 20% test set."
        )
        supervised_df = pd.read_csv(supervised_csv)
        st.dataframe(supervised_df, use_container_width=True, hide_index=True)

        best_row = supervised_df.loc[supervised_df["Accuracy"].idxmax()]
        st.success(
            f"🏆 Best performing model on this run: **{best_row['Model']}** "
            f"with **{best_row['Accuracy']*100:.2f}%** accuracy "
            "(determined automatically from the results above)."
        )

        if os.path.exists(chart_png):
            st.image(chart_png, caption="Accuracy comparison across all models")

        if os.path.exists(ann_curves_png):
            st.markdown("### ANN Training Curves")
            st.image(
                ann_curves_png,
                caption="ANN training/validation accuracy and loss over epochs",
            )
        else:
            st.info(
                "ℹ️ ANN training curves not found. This means TensorFlow was "
                "not installed when `train_all_models.py` was run. Install it "
                "with `pip install tensorflow` and re-run that script to "
                "include ANN results."
            )

        st.markdown("### Unsupervised Learning (Clustering)")
        st.caption(
            "These models were trained WITHOUT the FAKE/REAL labels. "
            "Silhouette Score measures how well-separated the clusters are "
            "using only the features. Adjusted Rand Index (ARI) compares the "
            "clusters to the true labels, but ONLY for evaluation afterwards "
            "- it was never used during clustering."
        )
        if os.path.exists(clustering_csv):
            clustering_df = pd.read_csv(clustering_csv)
            st.dataframe(clustering_df, use_container_width=True, hide_index=True)
        else:
            st.info("Clustering results not found.")

        if os.path.exists(reports_txt):
            with st.expander("See full classification reports & confusion matrices"):
                with open(reports_txt, "r", encoding="utf-8") as f:
                    st.text(f.read())
