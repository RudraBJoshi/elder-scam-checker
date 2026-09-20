"""
Train the elder-scam-detector model and save it to disk for the Flask app to load.

Built on the project's baseline_classifier.py approach (TF-IDF + Logistic
Regression -- the same classification technique from Rudra's Stanford/
DeepLearning.AI Supervised Machine Learning course), extended with char
n-grams and structural features after compare_models.py showed this
combination beats plain word TF-IDF on the validation set (see that file's
docstring for the comparison). Saves the fitted vectorizers + classifier as
.pkl files so app.py can load them at request time without retraining.

Usage:
    python model/train_model.py

Expects elder_focus_train.csv / elder_focus_val.csv / elder_focus_test.csv in
../data/ (relative to this file) -- run data/rebuild_dataset.py first if you've
changed the source data (see that file and data/fetch_real_smishing.py).
"""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    from model.feature_utils import build_features
except ImportError:
    from feature_utils import build_features

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE.parent / "data"
MODEL_DIR = HERE
FORMAL_LEGIT_WEIGHT = 8.0


def main():
    train = pd.read_csv(DATA_DIR / "elder_focus_train.csv")
    val = pd.read_csv(DATA_DIR / "elder_focus_val.csv")
    test = pd.read_csv(DATA_DIR / "elder_focus_test.csv")

    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    char_vec = TfidfVectorizer(max_features=3000, analyzer="char_wb", ngram_range=(3, 5), min_df=3)

    word_vec.fit(train["text"])
    char_vec.fit(train["text"])

    X_train = build_features(word_vec, char_vec, train["text"])
    X_val = build_features(word_vec, char_vec, val["text"])
    X_test = build_features(word_vec, char_vec, test["text"])

    # Formal-but-legitimate notices (bank alerts, appointment reminders, bills) share
    # vocabulary with phishing, and there are few of them relative to the rest of the
    # data. Upweighting them cut validation false alarms on this class from 22/40 to
    # 5/40 without adding missed scams (see README, "Accuracy improvements").
    sample_weight = np.where(train["scam_type"].eq("legitimate_formal"), FORMAL_LEGIT_WEIGHT, 1.0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=3)
    clf.fit(X_train, train["label"], sample_weight=sample_weight)

    metrics = {}
    for name, X, y in [("validation", X_val, val["label"]), ("test", X_test, test["label"])]:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        metrics[name] = {
            "accuracy": round(float(acc), 4),
            "n_rows": len(y),
        }
        print(f"=== {name.upper()} SET ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y, pred, target_names=["legit", "scam"]))
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y, pred))
        print()

    joblib.dump(word_vec, MODEL_DIR / "word_vectorizer.pkl")
    joblib.dump(char_vec, MODEL_DIR / "char_vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "classifier.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved word_vectorizer.pkl, char_vectorizer.pkl, classifier.pkl, "
          f"metrics.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()
