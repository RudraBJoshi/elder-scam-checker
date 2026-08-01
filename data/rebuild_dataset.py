"""
Rebuild the elder-focus train/val/test split with two fixes over the
original split_dataset.py (from the project's dataset docs):

1. Replaces the old 178-row `synthetic_ic3_grounded` set with the expanded,
   more diverse set from generate_elder_examples.py (see that file's
   docstring for why).
2. Splits with GROUPS instead of a plain per-row stratified split. The old
   split let near-duplicate rows from the same template (e.g. the same
   grandparent-scam template with just the dollar amount swapped) land on
   both sides of train/test -- an error-analysis pass found 12 of 16
   synthetic test rows were template-duplicates of a train row. Every
   synthetic row now carries a template_id; non-synthetic rows get a unique
   per-row group (so they still split independently, same as before) --
   sklearn's StratifiedGroupKFold then guarantees no template_id crosses a
   split boundary, while still balancing the label distribution.

Usage:
    python data/rebuild_dataset.py

Reads:
    elder_scam_dataset_full.csv    (the original 40,453-row full corpus)
    elder_synthetic_examples.csv   (freshly regenerated, with template_id)
    real_smishing_examples.csv     (real reported smishing, see fetch_real_smishing.py;
                                     optional -- skipped with a warning if not present)
Writes:
    elder_focus_train.csv / elder_focus_val.csv / elder_focus_test.csv
"""
import pathlib

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = pathlib.Path(__file__).parent


def main():
    full = pd.read_csv(HERE / "elder_scam_dataset_full.csv")
    synthetic = pd.read_csv(HERE / "elder_synthetic_examples.csv")

    # Drop the old synthetic rows from the full corpus; the new set replaces them.
    full = full[full["source"] != "synthetic_ic3_grounded"].copy()

    # Give every non-synthetic row its own unique group so it splits
    # independently, just like a normal per-row split would.
    full["template_id"] = [f"row::{i}" for i in full.index]

    frames = [full, synthetic]
    real_path = HERE / "real_smishing_examples.csv"
    if real_path.exists():
        real = pd.read_csv(real_path)
        frames.append(real)
        print(f"Including {len(real)} real-smishing rows from {real_path.name}")
    else:
        print(f"NOTE: {real_path.name} not found -- run fetch_real_smishing.py first "
              f"to include real reported smishing data. Continuing without it.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Combined full corpus: {len(combined)} rows "
          f"({combined['label'].value_counts().to_dict()})")

    # --- Elder-focused subset: SMS ham/spam + elder_targeted categories ---
    elder_mask = (
        (combined["source"] == "uci_sms_spam_collection") |
        (combined["elder_targeted"] == 1)
    )
    elder_df = combined[elder_mask].copy().reset_index(drop=True)
    print(f"\nElder-focused subset: {len(elder_df)} rows")
    print(elder_df["label"].value_counts())
    print(elder_df["scam_type"].value_counts())

    # --- Grouped, stratified 80/10/10 split ---
    # Stage 1: split off ~20% (val+test) using 5 folds, keep fold 0 as holdout.
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, holdout_idx = next(sgkf1.split(elder_df, elder_df["label"], elder_df["template_id"]))
    train_df = elder_df.iloc[train_idx].reset_index(drop=True)
    holdout_df = elder_df.iloc[holdout_idx].reset_index(drop=True)

    # Stage 2: split the holdout 50/50 into val/test, still grouped.
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(sgkf2.split(holdout_df, holdout_df["label"], holdout_df["template_id"]))
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)

    print(f"\ntrain: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # Sanity check: no template_id should appear in more than one split.
    train_groups = set(train_df["template_id"])
    val_groups = set(val_df["template_id"])
    test_groups = set(test_df["template_id"])
    assert not (train_groups & val_groups), "train/val group leakage!"
    assert not (train_groups & test_groups), "train/test group leakage!"
    assert not (val_groups & test_groups), "val/test group leakage!"
    print("No template_id shared across splits: confirmed")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name} label balance: {df['label'].value_counts().to_dict()}")

    train_df.to_csv(HERE / "elder_focus_train.csv", index=False)
    val_df.to_csv(HERE / "elder_focus_val.csv", index=False)
    test_df.to_csv(HERE / "elder_focus_test.csv", index=False)
    print("\nSaved elder_focus_train.csv / elder_focus_val.csv / elder_focus_test.csv")


if __name__ == "__main__":
    main()
