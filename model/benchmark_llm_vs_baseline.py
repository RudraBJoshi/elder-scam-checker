"""
Head-to-head comparison: our shipped TF-IDF + Logistic Regression model
vs. a frontier LLM (Claude Haiku), on the SAME held-out test messages.

This answers "would an LLM actually do better?" with real numbers instead
of guessing -- accuracy, F1, per-message latency, and estimated dollar
cost, side by side.

Requires (only for this script -- the main app needs none of this):
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   (your own key, from console.anthropic.com)

Cost note: this calls a paid API. Default sample size is 150 messages
(stratified by label), which costs well under $0.25 and takes 1-2 minutes
with default concurrency. Use --full to run the entire test set (~1,018
rows, roughly $1 and ~10-15 minutes) once you're ready for final numbers.
The script prints the estimated cost of the sample size BEFORE spending
anything, and asks for confirmation unless you pass --yes.

Usage:
    python model/benchmark_llm_vs_baseline.py                 # 150-row sample, asks to confirm
    python model/benchmark_llm_vs_baseline.py --n 300 --yes   # bigger sample, no prompt
    python model/benchmark_llm_vs_baseline.py --full --yes    # entire test set
    python model/benchmark_llm_vs_baseline.py --csv results.csv
"""
import argparse
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from model.feature_utils import build_features
    from model.llm_detector import classify_message_with_llm, get_client, DEFAULT_MODEL
except ImportError:
    from feature_utils import build_features
    from llm_detector import classify_message_with_llm, get_client, DEFAULT_MODEL

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE.parent / "data"

# Claude Haiku 4.5 pricing as of this writing -- see platform.claude.com/docs/en/about-claude/pricing.
# Update these if pricing changes; the script always also prints actual token
# usage so you can recompute cost with current rates regardless.
PRICE_PER_M_INPUT = 1.00
PRICE_PER_M_OUTPUT = 5.00


def load_baseline_predictions(df):
    word_vec = joblib.load(HERE / "word_vectorizer.pkl")
    char_vec = joblib.load(HERE / "char_vectorizer.pkl")
    clf = joblib.load(HERE / "classifier.pkl")

    preds, latencies = [], []
    for text in df["text"]:
        start = time.time()
        X = build_features(word_vec, char_vec, [text])
        pred = int(clf.predict(X)[0])
        latencies.append(time.time() - start)
        preds.append(pred)
    return preds, latencies


def run_llm_predictions(df, model, workers):
    client = get_client()
    results = [None] * len(df)

    def work(i, text, sender):
        return i, classify_message_with_llm(client, text, sender=sender, model=model)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(work, i, row["text"], row.get("sender") if "sender" in df.columns else None)
            for i, (_, row) in enumerate(df.iterrows())
        ]
        done = 0
        for fut in as_completed(futures):
            i, result = fut.result()
            results[i] = result
            done += 1
            if done % 25 == 0 or done == len(df):
                print(f"  ...{done}/{len(df)} LLM calls done")

    return results


def stratified_sample(df, n, seed):
    if n >= len(df):
        return df
    frac = n / len(df)
    sampled = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="Sample size (stratified by label)")
    parser.add_argument("--full", action="store_true", help="Use the entire test set instead of a sample")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=5, help="Concurrent LLM API calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save per-message results")
    args = parser.parse_args()

    test_df = pd.read_csv(DATA_DIR / "elder_focus_test.csv")
    df = test_df if args.full else stratified_sample(test_df, args.n, args.seed)

    # Rough cost estimate before spending anything: ~300 input / ~120 output tokens/msg is typical.
    est_cost = len(df) * (300 * PRICE_PER_M_INPUT + 120 * PRICE_PER_M_OUTPUT) / 1_000_000
    print(f"About to classify {len(df)} messages with {args.model}.")
    print(f"Rough estimated cost: ${est_cost:.2f} (actual cost printed at the end from real token usage).")
    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted -- no API calls made.")
            return

    print("\nRunning baseline (local TF-IDF + Logistic Regression) model...")
    base_preds, base_latencies = load_baseline_predictions(df)

    print("\nRunning LLM classifications (this calls the Anthropic API)...")
    try:
        llm_results = run_llm_predictions(df, args.model, args.workers)
    except ImportError:
        print(
            "\nERROR: the `anthropic` package isn't installed. Run:\n"
            "    pip install anthropic\n"
            "then try again."
        )
        return
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return

    llm_preds = [r["label"] for r in llm_results]
    llm_latencies = [r["latency_sec"] for r in llm_results if r["latency_sec"] is not None]
    n_errors = sum(1 for r in llm_results if r["error"] is not None)
    total_input_tok = sum(r["usage"]["input_tokens"] for r in llm_results if r["usage"])
    total_output_tok = sum(r["usage"]["output_tokens"] for r in llm_results if r["usage"])
    actual_cost = (total_input_tok * PRICE_PER_M_INPUT + total_output_tok * PRICE_PER_M_OUTPUT) / 1_000_000

    y_true = df["label"].tolist()

    # Only score rows where the LLM actually returned a prediction (errors excluded from its own metrics)
    valid_idx = [i for i, r in enumerate(llm_results) if r["label"] is not None]
    y_true_valid = [y_true[i] for i in valid_idx]
    llm_preds_valid = [llm_preds[i] for i in valid_idx]

    print(f"\n{'=' * 70}")
    print(f"RESULTS on {len(df)} test messages ({sum(y_true)} scam / {len(y_true) - sum(y_true)} legit)")
    print(f"{'=' * 70}")

    print(f"\n--- Local model (TF-IDF + char n-grams + structural features + Logistic Regression) ---")
    print(f"Accuracy:  {accuracy_score(y_true, base_preds):.4f}")
    print(f"F1:        {f1_score(y_true, base_preds):.4f}")
    print(f"Precision: {precision_score(y_true, base_preds):.4f}")
    print(f"Recall:    {recall_score(y_true, base_preds):.4f}")
    print(f"Avg latency: {sum(base_latencies) / len(base_latencies) * 1000:.2f} ms/message")
    print(f"Cost: $0.00 (runs entirely locally, no API calls)")

    print(f"\n--- LLM ({args.model}) ---")
    if n_errors:
        print(f"WARNING: {n_errors}/{len(df)} calls failed after retries and were excluded from scoring.")
    print(f"Accuracy:  {accuracy_score(y_true_valid, llm_preds_valid):.4f}")
    print(f"F1:        {f1_score(y_true_valid, llm_preds_valid):.4f}")
    print(f"Precision: {precision_score(y_true_valid, llm_preds_valid):.4f}")
    print(f"Recall:    {recall_score(y_true_valid, llm_preds_valid):.4f}")
    if llm_latencies:
        print(f"Avg latency: {sum(llm_latencies) / len(llm_latencies) * 1000:.0f} ms/message "
              f"(network round-trip; requires internet)")
    print(f"Actual cost: ${actual_cost:.4f} for {len(df)} messages "
          f"(${actual_cost / len(df) * 1000:.3f} per 1,000 messages)")

    if args.csv:
        out = df.copy().reset_index(drop=True)
        out["true_label"] = y_true
        out["local_pred"] = base_preds
        out["llm_pred"] = llm_preds
        out["llm_risk_level"] = [r["risk_level"] for r in llm_results]
        out["llm_scam_type"] = [r["scam_type"] for r in llm_results]
        out["llm_reasoning"] = [r["reasoning"] for r in llm_results]
        out["llm_error"] = [r["error"] for r in llm_results]
        out.to_csv(args.csv, index=False)
        print(f"\nPer-message results saved to {args.csv}")


if __name__ == "__main__":
    main()
