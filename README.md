# 📰 Fake News Detection using Machine Learning

A beginner-friendly Machine Learning project that predicts whether a news
headline/article is **FAKE** or **REAL**, along with a confidence score —
using **TF-IDF + Logistic Regression** and a **Streamlit** web interface.

---

## 1. Project Structure

```
fake-news-detection/
│
├── dataset/
│   ├── Fake.csv          # (you provide) all FAKE news articles
│   ├── True.csv          # (you provide) all REAL news articles
│   └── news.csv          # auto-created by train.py (combined + labeled)
│
├── model/
│   ├── model.pkl          # trained Logistic Regression model (created by train.py)
│   └── vectorizer.pkl     # trained TF-IDF vectorizer (created by train.py)
│
├── results/                       # created by train_all_models.py (see Section 9)
│   ├── final_model_comparison.csv     # Accuracy/Precision/Recall/F1 for all supervised models + ANN
│   ├── clustering_comparison.csv      # Silhouette Score + ARI for K-Means/Agglomerative/DBSCAN
│   ├── classification_reports.txt     # full report + confusion matrix per model
│   ├── model_comparison_chart.png     # accuracy bar chart
│   └── ann_training_curves.png        # ANN accuracy/loss curves
│
├── train.py                # trains and saves the MAIN Logistic Regression model
│                            # (this is the one app.py's "Predict" tab uses)
├── train_all_models.py     # ASSIGNMENT EXTENSION: trains Naive Bayes, KNN,
│                            # Decision Tree, Random Forest, SVM, an ANN, and
│                            # 3 clustering algorithms; saves results/ for app.py
├── app.py                  # Streamlit web app (Predict tab + Model Comparison tab)
├── requirements.txt        # Python dependencies
└── README.md
```

A **small sample dataset** (`dataset/Fake.csv`, `dataset/True.csv`) is
already included so you can run the whole project immediately and see it
work. It's synthetic/templated data meant only to prove the pipeline works —
**replace it with a real dataset** (instructions below) before relying on
the accuracy numbers or using it for anything real.

---

## 2. How It Works (Big Picture)

```
Raw text  →  Clean text  →  TF-IDF numbers  →  Logistic Regression  →  FAKE / REAL + confidence %
```

1. **Text Cleaning** – lowercase everything, strip punctuation/numbers/URLs,
   remove "stopwords" (common filler words like *the, is, and*).
2. **TF-IDF Vectorization** – converts cleaned text into numeric features a
   model can understand, weighting words by how distinctive they are.
3. **Logistic Regression** – a simple, fast, easy-to-explain classifier that
   estimates the *probability* that a text is FAKE.
4. **Streamlit App** – a simple web UI where you paste text and get an
   instant prediction + confidence score.

---

## 3. Installation

Make sure you have **Python 3.8+** installed. Then, from inside the
`fake-news-detection/` folder, run:

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, nltk, streamlit, joblib,
matplotlib, seaborn.

> **Note:** The first time `train.py` or `app.py` runs, it will try to
> download NLTK's "stopwords" list automatically. If you have no internet
> access at that moment, the code automatically falls back to a small
> built-in stopword list, so it will still work.

---

## 4. Getting a Real Dataset

The included sample data is just for testing the pipeline. For a real,
usable model, download a real Fake News dataset. The most common
beginner-friendly option:

**Kaggle: "Fake and Real News Dataset"**
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Steps:
1. Download the dataset from the link above (you'll need a free Kaggle account).
2. It contains two files: `Fake.csv` and `True.csv`.
3. Replace the sample files in `dataset/Fake.csv` and `dataset/True.csv`
   with the downloaded ones (same filenames, same folder).
4. Re-run `python train.py` to retrain on the real data.

`train.py` automatically:
- adds `label = 1` to every row from `Fake.csv`
- adds `label = 0` to every row from `True.csv`
- combines `title + text` into one text column
- merges both files into a single shuffled dataset (`dataset/news.csv`)

If instead you already have a single combined CSV with `text` and `label`
columns, just save it as `dataset/news.csv` and `train.py` will use it
directly (it checks for `Fake.csv`/`True.csv` first, and falls back to
`news.csv`).

---

## 5. Training the Model

From the `fake-news-detection/` folder, run:

```bash
python train.py
```

This will:
- Load and combine the dataset
- Clean the text
- Split into training (80%) and testing (20%) sets
- Vectorize text with TF-IDF
- Train a Logistic Regression model
- Print **Accuracy, Precision, Recall, F1-score, and a Confusion Matrix**
- Save `model/model.pkl` and `model/vectorizer.pkl`

---

## 6. Running the Web App

Once training is done (the `model/` folder has `model.pkl` and
`vectorizer.pkl`), start the Streamlit app:

```bash
streamlit run app.py
```

This opens a browser window where you can:
- Paste a news headline or article
- Click **Predict**
- See **FAKE** or **REAL**, a **confidence percentage**, and a plain-English
  explanation of the result

---

## 7. Code Walkthrough (for Viva / Interview)

### Why Logistic Regression?
It's a simple linear classification algorithm that works very well for
text classification tasks like this one, is fast to train, and — unlike
deep learning models — is easy to explain: it learns a *weight* for each
word/n-gram indicating how strongly that word pushes a prediction toward
FAKE or REAL.

### Why TF-IDF?
Machine learning models need numbers, not words. **TF-IDF (Term
Frequency–Inverse Document Frequency)** converts text into a numeric
vector where:
- **Term Frequency (TF)** = how often a word appears in a given document.
- **Inverse Document Frequency (IDF)** = how rare that word is across *all*
  documents.
- Words that appear a lot in one article but rarely elsewhere (distinctive
  words) get a high score. Common words like "the" get a very low score.

### Why remove stopwords?
Words like *"the", "is", "and"* appear in almost every sentence regardless
of topic, so they add noise rather than useful signal for classification.
Removing them helps the model focus on words that actually differ between
FAKE and REAL news.

### Why split into train/test sets?
We train the model on 80% of the data and test it on the remaining 20% it
has **never seen**. This tells us how well the model would perform on new,
unseen news articles — not just how well it memorized the training data.

### What do Accuracy, Precision, Recall, and F1-score mean?
- **Accuracy** – % of all predictions that were correct.
- **Precision** – of all articles the model called FAKE, what % actually
  were FAKE (measures false alarms).
- **Recall** – of all articles that were actually FAKE, what % did the
  model correctly catch (measures missed fakes).
- **F1-score** – the harmonic mean of precision and recall — a single
  balanced score.
- **Confusion Matrix** – a table showing correct vs incorrect predictions
  broken down by class (REAL/FAKE), so you can see exactly what kind of
  mistakes the model makes.

### How does `app.py` reuse the trained model?
`train.py` saves two files using `joblib`:
- `model.pkl` — the trained Logistic Regression model
- `vectorizer.pkl` — the *fitted* TF-IDF vectorizer (it remembers the
  vocabulary/weights learned from the training data)

`app.py` loads both with `joblib.load()`. It's essential to reuse the
**same** vectorizer that was fit during training — a new one would produce
a different numeric representation and the model's predictions would be
meaningless.

### How is the confidence score calculated?
Logistic Regression doesn't just output a label — it outputs a
**probability** for each class via `model.predict_proba()`. We show the
probability of the predicted class as the "confidence" percentage.

---

## 8. Quick Command Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Replace dataset/Fake.csv and dataset/True.csv with a real dataset

# 3. Train the main model (Logistic Regression - powers the Predict tab)
python train.py

# 4. (Optional, for the assignment) Train ALL algorithms and generate
#    the comparison results shown in the Model Comparison tab
python train_all_models.py

# 5. Run the web app
streamlit run app.py
```

---

## 9. Assignment Extension: All Supervised + Unsupervised Algorithms + ANN

This section covers `train_all_models.py`, which was added to satisfy an
assignment requirement to apply **all Supervised and Unsupervised Learning
Algorithms, and ANN**, on the same dataset.

### What this does NOT change

- `train.py` is untouched. It still trains and saves the Logistic
  Regression model + TF-IDF vectorizer that power the **Predict** tab.
- The Predict tab's behavior is exactly the same as before.

### What was added

**`train_all_models.py`** — a new, separate script that:
1. Reuses the *exact same* data loading + text cleaning as `train.py`
   (imported directly from it — there is only one preprocessing pipeline
   in the whole project).
2. Splits the data into the same 80/20 train/test split
   (`random_state=42`, same as `train.py`).
3. Fits a fresh TF-IDF vectorizer **on the training text only** (the test
   text is only ever transformed, never used to fit anything — this avoids
   *data leakage*).
4. Trains and evaluates 6 **supervised** models on those TF-IDF features:
   Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest,
   and Linear SVM.
5. Reduces the TF-IDF features to a smaller dense representation using
   **TruncatedSVD** (similar to PCA, but works on sparse TF-IDF data).
   This reduced representation is reused for both the ANN and the
   clustering algorithms below, so the dimensionality-reduction step is
   only done once.
6. Trains a simple **ANN** (feed-forward neural network, Keras/TensorFlow)
   on the reduced features and evaluates it the same way as the other
   models.
7. Runs 3 **unsupervised** clustering algorithms — **K-Means**,
   **Agglomerative (Hierarchical) Clustering**, and **DBSCAN** — on the
   held-out test set's reduced features, *without* using the FAKE/REAL
   labels to train them. The true labels are used only afterwards, to
   measure how well the discovered clusters line up with FAKE/REAL
   (via **Adjusted Rand Index**), alongside the label-free **Silhouette
   Score**.
8. Saves everything (comparison tables, full classification reports,
   confusion matrices, and charts) into a `results/` folder.

**`app.py`** — got a new **"📊 Model Comparison"** tab (the existing
"🔍 Predict" tab and all its logic are unchanged) that reads the files in
`results/` and displays:
- A table of Accuracy/Precision/Recall/F1-score for all 6 supervised
  models + the ANN, plus which one performed best *on this run*
- The accuracy comparison bar chart
- The ANN's training/validation accuracy & loss curves
- A table of Silhouette Score + Adjusted Rand Index for the 3 clustering
  algorithms
- An expandable section with the full classification report + confusion
  matrix text for every model

### How to run it

```bash
pip install -r requirements.txt      # now also installs tensorflow
python train_all_models.py           # trains everything, saves results/
streamlit run app.py                 # open the new "Model Comparison" tab
```

`train_all_models.py` can be re-run any time (e.g., after swapping in the
full Kaggle dataset) to refresh the results shown in the app.

> **Note:** if TensorFlow isn't installed, the script prints a warning and
> **skips only the ANN step** — every other model still runs and saves its
> results normally, so the rest of the assignment still works.

### Viva notes for the new algorithms

- **Naive Bayes** — assumes word features are independent of each other
  given the class (a simplifying "naive" assumption). Despite being
  simple, it's a classic, fast, and surprisingly effective baseline for
  text classification.
- **KNN (K-Nearest Neighbors)** — classifies a new article by looking at
  the `k` most similar articles (by TF-IDF distance) in the training data
  and taking a majority vote. Simple to understand, but slower to predict
  with as the dataset grows.
- **Decision Tree** — learns a series of if/else rules on the TF-IDF
  features to split FAKE vs REAL. Easy to explain, but prone to
  overfitting on its own.
- **Random Forest** — trains *many* different decision trees on random
  subsets of the data/features and lets them vote together. This usually
  performs better and overfits less than a single Decision Tree.
- **SVM (Support Vector Machine)** — finds the best possible boundary
  (hyperplane) that separates FAKE from REAL articles with the widest
  possible margin. `LinearSVC` is used here because it's fast and works
  well on high-dimensional, sparse TF-IDF text data.
- **ANN (Artificial Neural Network)** — a small stack of Dense
  ("fully-connected") layers with ReLU activations and Dropout (to reduce
  overfitting), ending in a single sigmoid output that gives a FAKE
  probability between 0 and 1 — similar in spirit to Logistic Regression,
  but with hidden layers that can learn more complex, non-linear patterns.
- **K-Means** — picks `k` (here, 2) cluster centers and repeatedly assigns
  each article to its nearest center, then moves the centers to the
  average of their assigned points, until it stabilizes.
- **Agglomerative (Hierarchical) Clustering** — starts with every article
  as its own tiny cluster, then repeatedly merges the two closest
  clusters together until only `k` remain.
- **DBSCAN** — groups together points that are closely packed
  (within a distance `eps`, with at least `min_samples` neighbors), and
  labels points that don't belong to any dense region as "noise" (-1).
  Unlike K-Means/Agglomerative, it decides the number of clusters
  automatically and doesn't force every point into a cluster.
- **Silhouette Score** — a label-free way to judge clustering quality: for
  each point, it compares how close it is to points in its own cluster
  vs. the nearest other cluster. Ranges from -1 (bad) to 1 (well
  separated); it does **not** use the true FAKE/REAL labels.
- **Adjusted Rand Index (ARI)** — used *only for evaluation*, after
  clustering is done: it compares the cluster assignments to the true
  FAKE/REAL labels and measures agreement, corrected for chance. 1.0 =
  perfect match with the true labels, 0.0 = no better than random
  guessing. Because it needs the true labels, it can only be computed
  when labels are available (as here, for checking) — it's never used
  *during* clustering itself.
- **Why we don't declare a "best" model in the code** — the assignment
  asks not to assume which algorithm is best. `train_all_models.py`
  therefore just computes everyone's metrics and then finds the highest
  accuracy automatically (`comparison_df["Accuracy"].idxmax()`), so the
  answer is always based on the actual results, not a guess — and it may
  change depending on which dataset (sample vs. real) you use.
- **Avoiding data leakage** — the TF-IDF vectorizer and the TruncatedSVD
  step are both `fit()` only on the training data and only ever
  `transform()`-ed on the test data; the ANN's `validation_split` is also
  carved out of the training data, never the test set. This ensures every
  model's reported metrics reflect performance on truly unseen data.

---

## 10. Limitations (good to mention in a viva)

- The model only learns statistical word patterns from its training data —
  it does **not** verify facts, check sources, or understand real-world
  truth. It can be fooled by well-written fake news or flag legitimate but
  unusually-worded real news.
- Performance depends heavily on the quality and size of the training
  dataset — the included sample dataset is synthetic and only meant to
  demonstrate that the pipeline runs correctly end-to-end.
- This is a **classical ML / NLP** project (TF-IDF + Logistic Regression),
  intentionally kept simple and explainable — no deep learning or
  transformer models are used.
