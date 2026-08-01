"""
Error analysis for the elder-scam-detector model.

Runs the currently-trained vectorizer/classifier against the held-out
validation and test sets and surfaces the model's worst mistakes -- the
"confidently wrong" predictions -- grouped by scam_type and source, so you
can see exactly where the model is weak instead of just looking at a single
aggregate accuracy number.

This is the same kind of investigation that found the real-world false
positive on a legitimate healthcare-appointment message: manually testing
one message at a time will only ever catch what you happen to try. This
script surfaces the worst offenders across the whole held-out set at once.

Usage:
    python model/error_analysis.py                  # both val + test
    python model/error_analysis.py --set test        # test only
    python model/error_analysis.py --top 30          # show more rows
    python model/error_analysis.py --csv errors.csv  # also save full results
"""
import argparse
import pathlib

import joblib
import pandas as pd

try:
    from model.feature_utils import build_features
except ImportError:
    from feature_utils import build_features

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_model(model_dir):
    word_vec = joblib.load(model_dir / "word_vectorizer.pkl")
    char_vec = joblib.load(model_dir / "char_vectorizer.pkl")
    clf = joblib.load(model_dir / "classifier.pkl")
    return word_vec, char_vec, clf


def score(df, word_vec, char_vec, clf):
    X = build_features(word_vec, char_vec, df["text"])
    df = df.copy()
    df["pred_prob"] = clf.predict_proba(X)[:, 1]
    df["pred_label"] = (df["pred_prob"] >= 0.5).astype(int)
    df["correct"] = df["pred_label"] == df["label"]
    return df


def summarize(df, label):
    print(f"\n=== {label} ({len(df)} rows) ===")
    acc = df["correct"].mean()
    print(f"Accuracy: {acc:.4f}")

    print("\nAccuracy by scam_type:")
    grp = df.groupby("scam_type").agg(
        n=("correct", "size"), accuracy=("correct", "mean")
    ).sort_values("accuracy")
    print(grp.to_string())

    print("\nAccuracy by source:")
    grp = df.groupby("source").agg(
        n=("correct", "size"), accuracy=("correct", "mean")
    ).sort_values("accuracy")
    print(grp.to_string())


def show_worst(df, top_n):
    fp = df[(df["label"] == 0) & (df["pred_label"] == 1)].sort_values("pred_prob", ascending=False)
    fn = df[(df["label"] == 1) & (df["pred_label"] == 0)].sort_values("pred_prob")

    print(f"\n--- Worst FALSE POSITIVES (legit called scam), top {top_n} ---")
    if fp.empty:
        print("(none)")
    for _, row in fp.head(top_n).iterrows():
        print(f"[{row['pred_prob']:.2f}] ({row['scam_type']}/{row['source']}) {row['text'][:140]!r}")

    print(f"\n--- Worst FALSE NEGATIVES (scam called legit), top {top_n} ---")
    if fn.empty:
        print("(none)")
    for _, row in fn.head(top_n).iterrows():
        print(f"[{row['pred_prob']:.2f}] ({row['scam_type']}/{row['source']}) {row['text'][:140]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["val", "test", "both"], default="both")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save full scored results")
    parser.add_argument("--model-dir", type=str, default=str(HERE))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    data_dir = pathlib.Path(args.data_dir)
    word_vec, char_vec, clf = load_model(model_dir)

    frames = []
    if args.set in ("val", "both"):
        val = score(pd.read_csv(data_dir / "elder_focus_val.csv"), word_vec, char_vec, clf)
        val["split"] = "val"
        frames.append(val)
    if args.set in ("test", "both"):
        test = score(pd.read_csv(data_dir / "elder_focus_test.csv"), word_vec, char_vec, clf)
        test["split"] = "test"
        frames.append(test)

    combined = pd.concat(frames, ignore_index=True)
    summarize(combined, "COMBINED HELD-OUT SET" if len(frames) > 1 else args.set.upper())
    show_worst(combined, args.top)

    if args.csv:
        combined.sort_values("pred_prob", ascending=False).to_csv(args.csv, index=False)
        print(f"\nFull scored results saved to {args.csv}")


if __name__ == "__main__":
    main()
