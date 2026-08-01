"""
Pull in real, professionally-collected smishing reports to supplement the
synthetic elder-scam examples with genuine phrasing variety.

Source: reportsmishing/Smishing-Dataset-IMC25 (github.com/reportsmishing/
Smishing-Dataset-IMC25), the labeled dataset behind "Fishing for Smishing:
Understanding SMS Phishing Infrastructure and Strategies by Mining Public
User Reports" (ACM IMC 2025). Licensed CC-BY-4.0 -- reuse is permitted with
attribution (see ATTRIBUTION below). 33,869 English-language rows out of
~48k total, already anonymized by the dataset's authors (phone numbers,
URLs, dates, and named entities are replaced with tokens like <URL>,
<PHONE_NUMBER>, <DATE_TIME>, <NAMED_ENTITY>).

Why this data, specifically: real reported smishing gives the model genuine
phrasing it hasn't seen from our own templates, and three of the dataset's
categories map directly onto documented elder-fraud vectors:
  - "hey mum/dad"  -> a real-world family-impersonation scam (the "Hi Mum"
    WhatsApp scam), i.e. exactly our grandparent_scam category.
  - "government"   -> government agency impersonation (IRS/SSA-style, plus
    international examples like fake NHS covid-test-kit phishing).
  - "delivery"      -> a category we didn't have at all: fake package/toll
    delivery-fee smishing, a well-documented elder-targeting vector.
  - "banking" / "telecom" -> account-lock / bill-related phishing, folded
    into our existing phishing_spoofing category.
"wrong number" is excluded (not clearly malicious); "spam"/"others" are
excluded as too ambiguous to label confidently.

The anonymization tokens are replaced with representative realistic
placeholders before use, so the model learns from realistic wording instead
of literal tokens like "<URL>" that would never appear in a real message a
user pastes into the app.

ATTRIBUTION (required by CC-BY-4.0): cite the IMC 2025 paper "Fishing for
Smishing: Understanding SMS Phishing Infrastructure and Strategies by Mining
Public User Reports" and the repository
https://github.com/reportsmishing/Smishing-Dataset-IMC25 wherever this data
or a model trained on it is described (this project's README does).

Usage:
    python data/fetch_real_smishing.py
Requires internet access; downloads directly from the GitHub repo above.
Writes: real_smishing_examples.csv
"""
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent
SOURCE_URL = "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv"

# Realistic stand-ins for the dataset's anonymization tokens.
TOKEN_REPLACEMENTS = {
    "<URL>": "http://bit.ly/3xK9fQ2",
    "<PHONE_NUMBER>": "1-800-555-0199",
    "<US_BANK_NUMBER>": "00123456",
    "<US_DRIVER_LICENSE>": "D1234567",
    "<DATE_TIME>": "05/12",
    "<NAMED_ENTITY>": "your provider",
    "<NRP>": "our team",
    "<LOCATION>": "your area",
    "<IP_ADDRESS>": "192.0.2.1",
}

CATEGORY_MAP = {
    "hey mum/dad": "grandparent_scam",
    "government": "government_impersonation",
    "delivery": "package_delivery_scam",
    "banking": "phishing_spoofing",
    "telecom": "phishing_spoofing",
}

# Cap how many rows we pull per category so no single category (banking has
# 10k+ English rows) swamps the rest of the dataset.
CATEGORY_CAPS = {
    "hey mum/dad": 10_000,   # take all of them, tiny category
    "government": 300,
    "delivery": 300,
    "banking": 400,
    "telecom": 150,
}


def clean_text(text):
    for token, replacement in TOKEN_REPLACEMENTS.items():
        text = text.replace(token, replacement)
    # Collapse any leftover unknown <TOKEN> patterns and stray pipe separators.
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = text.replace("|", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    print(f"Downloading {SOURCE_URL} ...")
    df = pd.read_csv(SOURCE_URL, on_bad_lines="skip")
    eng = df[df["language"] == "English"].copy()
    print(f"English rows: {len(eng)}")

    rows = []
    for raw_cat, our_cat in CATEGORY_MAP.items():
        subset = eng[eng["scam_type"] == raw_cat].copy()
        cap = CATEGORY_CAPS.get(raw_cat, 200)
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=42)
        for i, row in enumerate(subset.itertuples()):
            text = clean_text(str(row.text))
            if len(text) < 15:  # skip near-empty rows after cleaning
                continue
            rows.append({
                "text": text,
                "label": 1,
                "scam_type": our_cat,
                "elder_targeted": 1,
                "source": "reportsmishing_imc25",
                "template_id": f"real_smishing::{raw_cat}::{i}",  # each real row is its own group
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nFinal real-smishing rows: {len(out)}")
    print(out["scam_type"].value_counts())

    out.to_csv(HERE / "real_smishing_examples.csv", index=False)
    print("\nSaved real_smishing_examples.csv")


if __name__ == "__main__":
    main()
