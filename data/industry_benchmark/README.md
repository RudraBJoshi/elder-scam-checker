# Industry-standard benchmark dataset

`sms_spam_collection.csv` is derived from the **SMS Spam Collection**
dataset:

> Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI
> Machine Learning Repository. https://doi.org/10.24432/C5CC84

Licensed **CC BY 4.0** (free to share, adapt, and reuse for any purpose,
including derivative model training, with attribution). Original paper:
Almeida, Hidalgo & Yamakami, "Contributions to the Study of SMS Spam
Filtering: New Collection and Results," ACM DocEng 2011.

## What's in this file

5,169 unique English SMS messages (4,516 ham / 653 spam) after removing
exact-duplicate rows from the original 5,574-row corpus — a standard
preprocessing step also used in several published papers on this dataset.
Columns: `text`, `label` (1 = spam, 0 = ham).

`sms_spam_collection_raw.csv` is the unmodified source file as retrieved
(columns `v1`/`v2`), kept for provenance.

## Why this dataset

It's the de facto standard academic benchmark for SMS spam classification
— used as the basis for classic ML baselines (Naive Bayes, SVM) and
several deep-learning papers cited during this project's benchmark
research (see the main README's "ML benchmark research" section). Using it
lets us make a fair, literature-comparable claim about our modeling
*technique*, separate from our purpose-built elder-fraud dataset and model.

See `model/benchmark_industry_standard.py` for how it's used.
