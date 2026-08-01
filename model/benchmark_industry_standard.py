"""
Benchmark our modeling technique against a well-known academic dataset.

Why this exists: papers proposing new architectures (deep learning, agentic
LLMs, etc.) for scam/spam text classification almost never test on the same
data or task as this project (see README's "Benchmark research" notes), so
their headline numbers aren't a fair comparison. What *is* a fair, widely-
recognized comparison point is the SMS Spam Collection dataset (Almeida &
Hidalgo, 2011) -- the de facto standard academic benchmark for SMS spam
classification, cited in hundreds of papers, including as the likely basis
for many of the "modern" benchmark numbers floating around online.

This script runs two separate, clearly-labeled checks:

  1. INDUSTRY-STANDARD CHECK: fits our exact technique (word TF-IDF + char
     TF-IDF + structural features + Logistic Regression, from feature_utils.py
     / train_model.py) fresh on this dataset's own train split, and reports
     accuracy/F1 on its own held-out test split. This is the fair,
     apples-to-apples number: "how does our approach do on the same public
     benchmark the literature uses?"

  2. OUT-OF-DOMAIN CHECK (informational only, NOT a fair comparison): runs
     our actual production elder-scam model, trained only on our own
     elder-focused data, against this generic SMS spam dataset with zero
     retraining. This tells us how well our model generalizes to a
     different distribution (generic ads/spam vs. our elder-fraud-specific,
     narrative-style scam messages) -- expected to score lower, and that's
     fine; it is not the model we ship or claim credit for here.

Data: data/industry_benchmark/sms_spam_collection.csv (5,169 unique
messages after de-duplication; 4,516 ham / 653 spam), derived from the UCI
Machine Learning Repository "SMS Spam Collection" dataset:

    Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
    Licensed CC BY 4.0.

Usage:
    python model/benchmark_industry_standard.py
"""
import pathlib

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from model.feature_utils import build_features
except ImportError:
    from feature_utils import build_features

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE.parent / "data"
BENCH_PATH = DATA_DIR / "industry_benchmark" / "sms_spam_collection.csv"

# Representative literature numbers on this exact dataset / adjacent claims,
# for context only -- not re-derived by this script. See README /
# app_build_notes.md "Benchmark research" section for sourcing and caveats.
LITERATURE_CONTEXT = [
    ("Classic Naive Bayes + TF-IDF (literature range)", "~95-98% accuracy"),
    ("Classic SVM/Logistic Regression + TF-IDF (literature range)", "~97-99% accuracy"),
    ("CNN-GRU deep learning (Kaggle SMS spam variant, unconfirmed if same corpus)", "98.97% accuracy / 0.9596 F1"),
    ("ResNet deep learning (Kaggle SMS spam variant, unconfirmed if same corpus)", "99.08% accuracy / 0.9646 F1"),
]


def main():
    df = pd.read_csv(BENCH_PATH)
    print(f"Loaded {len(df)} rows from {BENCH_PATH.name} "
          f"({(df['label'] == 1).sum()} spam / {(df['label'] == 0).sum()} ham)")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # ---- Check 1: our technique, trained fresh on this dataset ----
    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    char_vec = TfidfVectorizer(max_features=3000, analyzer="char_wb", ngram_range=(3, 5), min_df=3)
    word_vec.fit(train_df["text"])
    char_vec.fit(train_df["text"])

    X_train = build_features(word_vec, char_vec, train_df["text"])
    X_test = build_features(word_vec, char_vec, test_df["text"])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_df["label"])
    pred = clf.predict(X_test)

    acc = accuracy_score(test_df["label"], pred)
    f1 = f1_score(test_df["label"], pred)

    print("\n=== CHECK 1: Our technique, trained + tested on the SMS Spam Collection dataset ===")
    print("(This is the fair, apples-to-apples comparison against literature.)")
    print(f"Accuracy: {acc:.4f}   F1: {f1:.4f}   (test set n={len(test_df)})")
    print(classification_report(test_df["label"], pred, target_names=["ham", "spam"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(test_df["label"], pred))

    # ---- Check 2: our actual production elder-scam model, zero-shot ----
    try:
        prod_word_vec = joblib.load(HERE / "word_vectorizer.pkl")
        prod_char_vec = joblib.load(HERE / "char_vectorizer.pkl")
        prod_clf = joblib.load(HERE / "classifier.pkl")

        X_all = build_features(prod_word_vec, prod_char_vec, df["text"])
        prod_pred = prod_clf.predict(X_all)
        prod_acc = accuracy_score(df["label"], prod_pred)
        prod_f1 = f1_score(df["label"], prod_pred)

        print("\n=== CHECK 2: Our production elder-scam model, zero-shot on this generic dataset ===")
        print("(Informational only -- NOT a fair comparison. Different domain/task: our model")
        print(" was trained on narrative elder-fraud messages, this dataset is generic SMS ads/spam.)")
        print(f"Accuracy: {prod_acc:.4f}   F1: {prod_f1:.4f}   (full dataset n={len(df)})")
        print(classification_report(df["label"], prod_pred, target_names=["ham", "spam"]))
    except FileNotFoundError:
        print("\n(Skipping Check 2 -- production model .pkl files not found in model/.)")

    print("\n=== Literature context (not computed by this script; see sourcing notes above) ===")
    for name, val in LITERATURE_CONTEXT:
        print(f"  {name}: {val}")

    print(
        "\nTakeaway: Check 1's number is the legitimate 'industry-standard-benchmark' "
        "comparison point -- it shows whether our word+char+structural TF-IDF + "
        "Logistic Regression technique is competitive with classic ML baselines on "
        "the same public dataset the literature uses. It is NOT a claim that our "
        "elder-scam model (a different, purpose-built model) was tested on this data."
    )


if __name__ == "__main__":
    main()
