"""
train.py
--------
This script trains a Fake News Detection model.

STEPS (in plain English):
1. Load the news data (Fake.csv + True.csv, or a single news.csv).
2. Combine title + text into one column, and create labels (1 = FAKE, 0 = REAL).
3. Clean the text (lowercase, remove punctuation/numbers, remove stopwords).
4. Split the data into training and testing sets.
5. Convert text into numbers using TF-IDF (machines can't read words, only numbers).
6. Train a Logistic Regression model on the numbers.
7. Check how good the model is (accuracy, precision, recall, F1-score).
8. Save the trained model and the TF-IDF vectorizer to disk so app.py can reuse them.

Run this file with:
    python train.py
"""

import os
import re
import string

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# STEP 0: Make sure NLTK stopwords are available
# ---------------------------------------------------------------------------
# NLTK needs a small "stopwords" data file (words like "the", "is", "and"
# that don't add much meaning). We try to download it once. If nltk is not
# installed, or there is no internet connection at run time, we fall back to
# a small built-in list so the script still works either way.
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

    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback list used if nltk is missing or its data could not be downloaded.
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "is", "are",
        "was", "were", "be", "been", "being", "to", "of", "in", "on",
        "for", "with", "as", "by", "at", "from", "this", "that", "it",
        "its", "he", "she", "they", "them", "his", "her", "their", "we",
        "you", "your", "i", "not", "so", "no", "do", "does", "did",
    }

DATASET_DIR = "dataset"
MODEL_DIR = "model"


# ---------------------------------------------------------------------------
# STEP 1: Load the dataset
# ---------------------------------------------------------------------------
def load_dataset():
    """
    Loads the dataset in one of two supported formats:

    A) Separate Fake.csv and True.csv files (the common Kaggle format):
       dataset/Fake.csv  -> all rows are FAKE news
       dataset/True.csv  -> all rows are REAL news

    B) A single combined file dataset/news.csv that already has
       a 'text' column and a 'label' column (0 = REAL, 1 = FAKE).

    Returns a pandas DataFrame with two columns: 'text' and 'label'.
    """
    fake_path = os.path.join(DATASET_DIR, "Fake.csv")
    true_path = os.path.join(DATASET_DIR, "True.csv")
    combined_path = os.path.join(DATASET_DIR, "news.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("Found Fake.csv and True.csv -> combining them...")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        # Label the data: FAKE = 1, REAL = 0
        fake_df["label"] = 1
        true_df["label"] = 0

        # Combine title + text into a single 'text' column if both exist.
        for df in (fake_df, true_df):
            if "title" in df.columns and "text" in df.columns:
                df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            elif "title" in df.columns and "text" not in df.columns:
                df["text"] = df["title"].fillna("")
            # if only 'text' exists, nothing to do

        fake_df = fake_df[["text", "label"]]
        true_df = true_df[["text", "label"]]

        # Combine both into one DataFrame and shuffle the rows.
        data = pd.concat([fake_df, true_df], ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save the combined file for reference / reuse.
        data.to_csv(combined_path, index=False)
        print(f"Combined dataset saved to {combined_path}")
        return data

    elif os.path.exists(combined_path):
        print("Found dataset/news.csv -> loading it directly...")
        data = pd.read_csv(combined_path)
        return data

    else:
        raise FileNotFoundError(
            "No dataset found. Please place either:\n"
            "  - dataset/Fake.csv and dataset/True.csv, OR\n"
            "  - dataset/news.csv (with 'text' and 'label' columns)\n"
            "See the README for download instructions."
        )


# ---------------------------------------------------------------------------
# STEP 2: Clean the text
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Cleans a single piece of text:
    - converts to lowercase
    - removes URLs
    - removes punctuation and numbers
    - removes extra whitespace
    - removes stopwords (common words with little meaning, e.g. 'the', 'is')
    """
    text = str(text).lower()                              # lowercase
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # remove URLs
    text = re.sub(r"<.*?>", " ", text)                     # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                # remove extra spaces

    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def load_and_clean_data():
    """
    Loads the dataset, handles missing values, and cleans the text.
    Returns a pandas DataFrame with (at least) two columns:
    'clean_text' and 'label'.

    NOTE: This function is the SINGLE source of truth for preprocessing in
    this project. Both train.py (Logistic Regression model used by the
    Streamlit app) and train_all_models.py (all the other ML algorithms
    added later for the assignment) call this same function, so there is
    only ONE preprocessing pipeline in the whole project - nothing is
    duplicated.
    """
    # STEP 1: Load data
    data = load_dataset()
    print(f"Total rows loaded: {len(data)}")

    # STEP 2 (part of Handling missing values): drop empty rows
    data = data.dropna(subset=["text", "label"])
    data = data[data["text"].str.strip() != ""]
    data["label"] = data["label"].astype(int)
    print(f"Rows remaining after removing missing/empty values: {len(data)}")

    # STEP 3: Clean all the text
    print("Cleaning text... (this may take a moment)")
    data["clean_text"] = data["text"].apply(clean_text)

    # Drop rows that became empty after cleaning
    data = data[data["clean_text"].str.strip() != ""]

    return data


def main():
    # -----------------------------------------------------------------
    # STEPS 1-3: Load + handle missing values + clean text
    # -----------------------------------------------------------------
    data = load_and_clean_data()

    # -----------------------------------------------------------------
    # STEP 4: Separate features (X) and labels (y)
    # -----------------------------------------------------------------
    X = data["clean_text"]
    y = data["label"]

    print("\nLabel distribution (0 = REAL, 1 = FAKE):")
    print(y.value_counts())

    # -----------------------------------------------------------------
    # STEP 5: Train/test split
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # -----------------------------------------------------------------
    # STEP 6: TF-IDF Vectorization
    # -----------------------------------------------------------------
    # TF-IDF turns text into numbers by scoring each word based on:
    #   - how often it appears in a document (Term Frequency)
    #   - how rare/common it is across all documents (Inverse Document Frequency)
    # Words that are common in ONE article but rare overall get a high score
    # because they are likely important/distinctive words for that article.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -----------------------------------------------------------------
    # STEP 7: Train Logistic Regression model
    # -----------------------------------------------------------------
    # Logistic Regression is a simple, fast, and easy-to-explain classification
    # algorithm. It estimates the probability that a piece of text is FAKE (1)
    # vs REAL (0), based on the TF-IDF word scores.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # -----------------------------------------------------------------
    # STEP 8 & 9: Evaluate the model
    # -----------------------------------------------------------------
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                Predicted REAL   Predicted FAKE")
    print(f"Actual REAL          {cm[0][0]:<10}     {cm[0][1]:<10}")
    print(f"Actual FAKE          {cm[1][0]:<10}     {cm[1][1]:<10}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # -----------------------------------------------------------------
    # STEP 10: Save the model and vectorizer
    # -----------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print("\nTraining complete! You can now run the Streamlit app with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
