"""
train_all_models.py
--------------------
ASSIGNMENT EXTENSION: "Apply all Supervised and Unsupervised Learning
Algorithms and ANN on the given dataset."

WHAT THIS FILE IS (and is NOT):
- This is an ADDITIONAL script. It does NOT replace train.py.
- train.py still trains/saves the Logistic Regression model + vectorizer
  that power the live "Predict" tab in the Streamlit app (app.py). That
  is untouched and keeps working exactly as before.
- This script trains several MORE algorithms (for the assignment
  requirement), evaluates all of them, and saves the results (tables,
  charts, text reports) into a results/ folder. app.py then reads those
  saved results and displays them on a new "Model Comparison" tab.

REUSED PREPROCESSING (no duplicate pipeline):
This script imports `load_and_clean_data` and `clean_text` directly from
train.py. That means the dataset loading, missing-value handling, and text
cleaning (lowercase, remove punctuation/numbers/URLs, remove stopwords) are
done in EXACTLY the same way as train.py - there is only one preprocessing
pipeline in the whole project.

ALGORITHMS COVERED:

SUPERVISED (trained WITH the FAKE/REAL labels, tested on unseen data):
  - Logistic Regression
  - Naive Bayes (Multinomial)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (Linear SVM)

ANN (simple feed-forward neural network, Keras/TensorFlow):
  - A small Dense neural network, evaluated the same way as the
    supervised models above.

UNSUPERVISED (trained WITHOUT labels - labels are used ONLY afterwards,
to check how well the discovered groups line up with FAKE/REAL):
  - K-Means
  - Agglomerative (Hierarchical) Clustering
  - DBSCAN

Run with:
    python train_all_models.py

This creates a results/ folder containing:
  - final_model_comparison.csv   (Logistic Regression, Naive Bayes, KNN,
                                   Decision Tree, Random Forest, SVM, ANN)
  - clustering_comparison.csv    (K-Means, Agglomerative, DBSCAN)
  - classification_reports.txt   (full report + confusion matrix per model)
  - model_comparison_chart.png   (bar chart comparing accuracy)
  - ann_training_curves.png      (ANN training/validation accuracy & loss)
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display (server-friendly)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    adjusted_rand_score,
)

# Reuse the EXISTING preprocessing from train.py - do not duplicate it.
from train import load_and_clean_data

RANDOM_STATE = 42          # used everywhere for reproducible results
TEST_SIZE = 0.20           # same split ratio as train.py
MAX_FEATURES = 5000        # same TF-IDF size as train.py

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# This text file will collect the full classification_report() and
# confusion matrix for every classification model, so results are easy
# to copy into a viva/report.
report_lines = []


def log(text=""):
    """Prints to the console AND stores the line for classification_reports.txt"""
    print(text)
    report_lines.append(str(text))


def evaluate_classifier(name, y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall, F1-score, confusion matrix and
    the full classification_report for one model's predictions, prints
    them, and returns a summary dict (used to build the comparison table).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    log("\n" + "=" * 60)
    log(f"MODEL: {name}")
    log("=" * 60)
    log(f"Accuracy : {acc:.4f}")
    log(f"Precision: {prec:.4f}")
    log(f"Recall   : {rec:.4f}")
    log(f"F1-Score : {f1:.4f}")
    log("\nConfusion Matrix:")
    log("                Predicted REAL   Predicted FAKE")
    log(f"Actual REAL          {cm[0][0]:<10}     {cm[0][1]:<10}")
    log(f"Actual FAKE          {cm[1][0]:<10}     {cm[1][1]:<10}")
    log("\nClassification Report:")
    log(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], zero_division=0))

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    }


def main():
    overall_start = time.time()

    # -----------------------------------------------------------------
    # STEP 1: Load + clean data using the SAME function as train.py
    # -----------------------------------------------------------------
    data = load_and_clean_data()
    X = data["clean_text"]
    y = data["label"]

    log("Label distribution (0 = REAL, 1 = FAKE):")
    log(str(y.value_counts()))

    # -----------------------------------------------------------------
    # STEP 2: Train/test split (same ratio & random_state as train.py)
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # -----------------------------------------------------------------
    # STEP 3: TF-IDF vectorization (same configuration as train.py)
    # -----------------------------------------------------------------
    # IMPORTANT (avoiding data leakage): the vectorizer is FIT only on the
    # training text. The test text is only ever TRANSFORMED, never used to
    # fit the vectorizer - so no information from the test set leaks into
    # training.
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # =========================================================================
    # PART A: SUPERVISED LEARNING MODELS
    # =========================================================================
    log("\n\n" + "#" * 70)
    log("# PART A: SUPERVISED LEARNING MODELS")
    log("#" * 70)

    # A dictionary of model name -> untrained model. Looping over this
    # avoids writing near-identical training code six separate times.
    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (Linear)": LinearSVC(random_state=RANDOM_STATE),
    }

    comparison_rows = []

    for name, model in supervised_models.items():
        log(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        result = evaluate_classifier(name, y_test, y_pred)
        result["Training Time (s)"] = round(time.time() - t0, 2)
        comparison_rows.append(result)

    # =========================================================================
    # PART B: DIMENSIONALITY REDUCTION (shared by ANN + clustering below)
    # =========================================================================
    # TF-IDF gives very high-dimensional, sparse vectors (up to MAX_FEATURES
    # columns). Neural networks need dense input, and distance-based
    # clustering algorithms (K-Means, Agglomerative, DBSCAN) work poorly in
    # very high dimensions ("curse of dimensionality"). TruncatedSVD (a
    # technique similar to PCA, but works directly on sparse TF-IDF data) is
    # used ONCE here to create a smaller, dense representation, and that
    # SAME reduced representation is reused for both the ANN and the
    # clustering algorithms below - so we don't repeat this step twice.
    n_components = min(200, X_train_tfidf.shape[0] - 1, X_train_tfidf.shape[1] - 1)
    log(f"\n\nReducing TF-IDF features to {n_components} dimensions using TruncatedSVD "
        f"(shared by the ANN and the clustering algorithms below)...")

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_dense = svd.fit_transform(X_train_tfidf)   # fit ONLY on training data
    X_test_dense = svd.transform(X_test_tfidf)          # test data only transformed

    # =========================================================================
    # PART C: ARTIFICIAL NEURAL NETWORK (ANN)
    # =========================================================================
    log("\n\n" + "#" * 70)
    log("# PART C: ARTIFICIAL NEURAL NETWORK (ANN)")
    log("#" * 70)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(RANDOM_STATE)

        # A simple, beginner-friendly feed-forward (Dense) neural network:
        #   Input -> Dense(128, relu) -> Dropout -> Dense(64, relu) -> Dense(1, sigmoid)
        # "sigmoid" output gives a probability between 0 and 1 (FAKE vs REAL),
        # similar in spirit to Logistic Regression, but the hidden layers let
        # it learn more complex, non-linear patterns.
        ann_model = keras.Sequential([
            layers.Input(shape=(X_train_dense.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])

        ann_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        log(ann_model.summary())

        t0 = time.time()
        history = ann_model.fit(
            X_train_dense, y_train,
            validation_split=0.2,   # carved out of the TRAINING data only
            epochs=15,
            batch_size=32,
            verbose=2,
        )
        ann_train_time = round(time.time() - t0, 2)

        # Predict on the untouched test set
        ann_probs = ann_model.predict(X_test_dense).ravel()
        ann_pred = (ann_probs >= 0.5).astype(int)

        ann_result = evaluate_classifier("ANN (Neural Network)", y_test, ann_pred)
        ann_result["Training Time (s)"] = ann_train_time
        comparison_rows.append(ann_result)

        # ---- Plot training/validation accuracy & loss curves ----
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(history.history["accuracy"], label="Training Accuracy")
        axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
        axes[0].set_title("ANN Accuracy over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(history.history["loss"], label="Training Loss")
        axes[1].plot(history.history["val_loss"], label="Validation Loss")
        axes[1].set_title("ANN Loss over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        curves_path = os.path.join(RESULTS_DIR, "ann_training_curves.png")
        plt.savefig(curves_path, dpi=120)
        plt.close(fig)
        log(f"\nSaved ANN training curves to {curves_path}")

    except ImportError:
        log(
            "\n⚠️  TensorFlow is not installed, so the ANN step was skipped.\n"
            "Install it with:  pip install tensorflow\n"
            "then re-run:      python train_all_models.py"
        )

    # =========================================================================
    # PART D: FINAL SUPERVISED + ANN COMPARISON TABLE
    # =========================================================================
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)"]
    ]

    log("\n\n" + "#" * 70)
    log("# FINAL COMPARISON: SUPERVISED MODELS + ANN")
    log("#" * 70)
    log(comparison_df.to_string(index=False))

    best_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    log(f"\nBest performing model on this run (highest accuracy): "
        f"{best_row['Model']} ({best_row['Accuracy']:.4f})")
    log("(Note: this is determined automatically from the results above, "
        "not assumed in advance - it may change depending on the dataset used.)")

    comparison_csv_path = os.path.join(RESULTS_DIR, "final_model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    log(f"\nSaved comparison table to {comparison_csv_path}")

    # Bar chart comparing accuracy across all models
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(comparison_df["Model"], comparison_df["Accuracy"], color="#4C72B0")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison (Supervised + ANN)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=120)
    plt.close(fig)
    log(f"Saved comparison chart to {chart_path}")

    # =========================================================================
    # PART E: UNSUPERVISED LEARNING (CLUSTERING)
    # =========================================================================
    # Clustering algorithms are given ONLY the features (X). They never see
    # the FAKE/REAL labels while grouping the data. We use the held-out
    # test set's reduced features (X_test_dense) for clustering, and the
    # matching true labels (y_test) are used AFTERWARDS, only to measure how
    # well the clusters happen to line up with FAKE/REAL - this is normal
    # practice for evaluating clustering quality, not "training with labels".
    log("\n\n" + "#" * 70)
    log("# PART E: UNSUPERVISED LEARNING (CLUSTERING)")
    log("#" * 70)

    X_cluster = X_test_dense
    y_cluster_true = y_test.values

    clustering_rows = []

    # ---- K-Means ----
    log("\nRunning K-Means (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_cluster)
    clustering_rows.append(("K-Means", kmeans_labels))

    # ---- Agglomerative (Hierarchical) Clustering ----
    log("Running Agglomerative (Hierarchical) Clustering (k=2)...")
    agglo = AgglomerativeClustering(n_clusters=2)
    agglo_labels = agglo.fit_predict(X_cluster)
    clustering_rows.append(("Agglomerative Clustering", agglo_labels))

    # ---- DBSCAN ----
    # DBSCAN needs an "eps" (neighborhood radius) parameter. Instead of
    # guessing a fixed number, we estimate a reasonable eps using a simple,
    # common heuristic: the median distance from each point to its 5th
    # nearest neighbor.
    log("Estimating a suitable eps for DBSCAN...")
    k = min(5, len(X_cluster) - 1)
    neighbors_model = NearestNeighbors(n_neighbors=k)
    neighbors_model.fit(X_cluster)
    distances, _ = neighbors_model.kneighbors(X_cluster)
    eps_estimate = float(np.median(distances[:, -1]))
    log(f"Using eps = {eps_estimate:.4f}, min_samples = {k}")

    dbscan = DBSCAN(eps=eps_estimate, min_samples=k)
    dbscan_labels = dbscan.fit_predict(X_cluster)
    clustering_rows.append(("DBSCAN", dbscan_labels))

    log("\n" + "-" * 60)
    log("CLUSTERING RESULTS (Silhouette Score & Adjusted Rand Index)")
    log("-" * 60)
    log("Silhouette Score: measures how well-separated the clusters are,")
    log("using ONLY the features (ranges -1 to 1, higher = better defined).")
    log("Adjusted Rand Index (ARI): compares clusters to the TRUE FAKE/REAL")
    log("labels, ONLY for evaluation (ranges -1 to 1, higher = closer match,")
    log("0 = random, 1 = perfect match). ARI is not used to train the model.")

    clustering_summary = []
    for name, labels in clustering_rows:
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_found >= 2:
            sil = silhouette_score(X_cluster, labels)
        else:
            sil = np.nan
            log(f"\n{name}: could not compute silhouette score "
                f"(fewer than 2 real clusters were found).")

        ari = adjusted_rand_score(y_cluster_true, labels)

        log(f"\n{name}:")
        log(f"  Clusters found     : {n_clusters_found}"
            f"{' (+ noise points)' if -1 in labels else ''}")
        log(f"  Silhouette Score   : {sil:.4f}" if not np.isnan(sil) else "  Silhouette Score   : N/A")
        log(f"  Adjusted Rand Index: {ari:.4f}")

        clustering_summary.append({
            "Model": name,
            "Clusters Found": n_clusters_found,
            "Silhouette Score": round(sil, 4) if not np.isnan(sil) else "N/A",
            "Adjusted Rand Index (ARI)": round(ari, 4),
        })

    clustering_df = pd.DataFrame(clustering_summary)
    clustering_csv_path = os.path.join(RESULTS_DIR, "clustering_comparison.csv")
    clustering_df.to_csv(clustering_csv_path, index=False)
    log(f"\nSaved clustering comparison table to {clustering_csv_path}")

    # =========================================================================
    # Save the full text log (all reports + confusion matrices)
    # =========================================================================
    reports_path = os.path.join(RESULTS_DIR, "classification_reports.txt")
    with open(reports_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nSaved full classification reports to {reports_path}")

    total_time = round(time.time() - overall_start, 2)
    log(f"\nAll done in {total_time} seconds.")
    log("You can now run:  streamlit run app.py  and open the 'Model Comparison' tab.")


if __name__ == "__main__":
    main()
