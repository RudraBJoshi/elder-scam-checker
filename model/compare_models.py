"""
Compare candidate feature/model configurations on the validation set, to
decide what train_model.py should actually use in production.

Candidates tried:
  1. baseline       - word TF-IDF (1,2-grams) + Logistic Regression (current production config)
  2. word_char      - word TF-IDF + char TF-IDF (catches spelling tricks like "arnaz0n") + Logistic Regression
  3. word_structural - word TF-IDF + hand-picked structural features (message length,
                       dollar-sign density, exclamation density, all-caps word ratio,
                       URL-like token presence) + Logistic Regression
  4. linear_svm     - word TF-IDF + calibrated Linear SVM (SVMs are often strong on
                       sparse TF-IDF text; CalibratedClassifierCV gives us predict_proba
                       back, which the app's risk tiers depend on)
  5. combined       - word+char TF-IDF + structural features + Logistic Regression

Model selection happens ONLY on the validation set. The test set stays
untouched until train_model.py evaluates the final chosen configuration, so
the reported test accuracy stays an honest, unbiased estimate.

Usage:
    python model/compare_models.py
"""
import pathlib
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def structural_features(texts):
    """Hand-picked signals that a pure word-TF-IDF model can't easily represent:
    how "shouty"/urgent-looking a message is, independent of its specific words."""
    rows = []
    for t in texts:
        t = str(t)
        length = len(t)
        n_dollar = t.count("$")
        n_exclaim = t.count("!")
        words = t.split()
        n_words = max(len(words), 1)
        allcaps_ratio = sum(1 for w in words if len(w) > 2 and w.isupper()) / n_words
        has_url = 1.0 if re.search(r"https?://|www\.|bit\.ly", t, re.I) else 0.0
        rows.append([
            length / 500.0,           # scaled roughly to [0, a few]
            n_dollar / n_words,
            n_exclaim / n_words,
            allcaps_ratio,
            has_url,
        ])
    return csr_matrix(np.array(rows, dtype=float))


def evaluate(name, X_train, y_train, X_val, y_val, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    print(f"{name:20s}  val_accuracy={acc:.4f}  val_f1={f1:.4f}")
    return acc, f1


def main():
    train = pd.read_csv(DATA_DIR / "elder_focus_train.csv")
    val = pd.read_csv(DATA_DIR / "elder_focus_val.csv")

    y_train, y_val = train["label"], val["label"]

    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    Xw_train = word_vec.fit_transform(train["text"])
    Xw_val = word_vec.transform(val["text"])

    char_vec = TfidfVectorizer(max_features=3000, analyzer="char_wb", ngram_range=(3, 5), min_df=3)
    Xc_train = char_vec.fit_transform(train["text"])
    Xc_val = char_vec.transform(val["text"])

    Xs_train = structural_features(train["text"])
    Xs_val = structural_features(val["text"])

    results = {}

    print("=== Candidate comparison (selected on VALIDATION only) ===")
    results["1_baseline"] = evaluate(
        "1_baseline (word+LR)", Xw_train, y_train, Xw_val, y_val,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    Xwc_train = hstack([Xw_train, Xc_train]).tocsr()
    Xwc_val = hstack([Xw_val, Xc_val]).tocsr()
    results["2_word_char"] = evaluate(
        "2_word_char (word+char+LR)", Xwc_train, y_train, Xwc_val, y_val,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    Xws_train = hstack([Xw_train, Xs_train]).tocsr()
    Xws_val = hstack([Xw_val, Xs_val]).tocsr()
    results["3_word_structural"] = evaluate(
        "3_word_structural (word+struct+LR)", Xws_train, y_train, Xws_val, y_val,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000), cv=3)
    results["4_linear_svm"] = evaluate(
        "4_linear_svm (word+SVM)", Xw_train, y_train, Xw_val, y_val, svm,
    )

    Xwcs_train = hstack([Xw_train, Xc_train, Xs_train]).tocsr()
    Xwcs_val = hstack([Xw_val, Xc_val, Xs_val]).tocsr()
    results["5_combined"] = evaluate(
        "5_combined (word+char+struct+LR)", Xwcs_train, y_train, Xwcs_val, y_val,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    print("\nBest by val accuracy:", max(results, key=lambda k: results[k][0]))
    print("Best by val F1:      ", max(results, key=lambda k: results[k][1]))


if __name__ == "__main__":
    main()
