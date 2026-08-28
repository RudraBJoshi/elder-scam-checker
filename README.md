# Elder Scam Message Checker

A senior-friendly web app for the 2026 Congressional App Challenge (CA-50). A
user pastes or types a suspicious text, email, or description of a phone call
and gets back a plain-language risk assessment: a traffic-light verdict, why
it was flagged, and what to do next.

## Project structure

```
elder_scam_app/
  app.py                     Flask routes: / (form), /check (analyze), /about
  model/
    scam_detector.py         ScamDetector: loads the model, predicts, explains
    feature_utils.py         Shared word+char+structural feature builder (train & inference)
    train_model.py           Trains + saves word_vectorizer.pkl / char_vectorizer.pkl / classifier.pkl
    compare_models.py        Compares feature/model configs on the validation set
    error_analysis.py        Surfaces the model's worst false positives/negatives
    word_vectorizer.pkl, char_vectorizer.pkl, classifier.pkl, metrics.json  (generated)
  data/
    generate_elder_examples.py    Regenerates the synthetic elder-scam examples (seeded)
    fetch_real_smishing.py        Pulls in real, CC-BY-licensed smishing reports (see below)
    rebuild_dataset.py             Merges everything + does a leakage-safe grouped split
    elder_focus_train/val/test.csv  The current dataset (8,142 / 1,018 / 1,018 rows)
  templates/
    base.html, index.html, _result.html, about.html
  static/
    css/style.css
    js/app.js
  requirements.txt
```

## Model status and training history

Current numbers, on a genuinely held-out, leakage-free test set:

- Validation accuracy: 97.4%
- Test accuracy: **97.5%** (precision/recall both 0.96-0.98 for both classes)

This looks similar to the project's original 97.7% baseline number, but it's
a different, harder, more honest measurement — getting here involved finding
and fixing three real problems along the way, in case you extend this
further:

**1. A real-world false positive, and its root cause.** Rudra tested a
genuine Sharp Healthcare appointment notification and the app called it a
SCAM (0.84 probability, zero matching red flags). Root cause, confirmed via
`error_analysis.py`: the original dataset's "legitimate" class was 100%
casual SMS-style text, with zero formal institutional notifications. Fix:
added a new `legitimate_formal` synthetic category (health/banking/delivery/
subscription notifications, `data/generate_elder_examples.py`), and changed
the risk-tiering logic so a "high risk / SCAM" verdict requires at least one
concrete, explainable red flag (a matched tactic, scam category, or
suspicious sender signal) — a raw model score alone, with nothing to point
to, now shows "we're not fully sure" instead of a false alarm
(`scam_detector.py`'s `_risk_tier`). This is a real, mostly-solved-by-data
fix (see the `legitimate_formal` caveat below) *plus* an architectural
safety net that keeps any residual model weakness from becoming a confident
false alarm.

**2. Train/test leakage in the hand-authored scam categories.** The
elder-scam examples are generated from templates with variables filled in
(names, dollar amounts). The original split put rows from the *same
template* on both sides of train/test (e.g. "$25,000" in train, "$2,500,000"
in test, otherwise identical wording) — an error-analysis check found 75% of
those test rows were template-duplicates of a train row, meaning the
headline accuracy on those categories was partly memorization, not
generalization. Fix: every synthetic row now carries a `template_id`, and
`data/rebuild_dataset.py` uses `StratifiedGroupKFold` to guarantee no
template ever crosses a split boundary. Test accuracy honestly *dropped*
after this fix (97.7% → 94.0%) before other improvements brought it back up
— that drop was the leakage being removed, not something to be alarmed by.

**3. Real-world data to replace thin synthetic categories.** Categories like
grandparent_scam and romance_scam only had 3-4 templates (about 20-40 rows)
each, which is too little for a linear model to generalize past its exact
training phrasing — a held-out check on the newly-independent (leakage-free)
splits showed romance_scam at 0% accuracy and grandparent_scam at 33%. Fixed
by pulling in real reported smishing from
[reportsmishing/Smishing-Dataset-IMC25](https://github.com/reportsmishing/Smishing-Dataset-IMC25)
(CC-BY-4.0; see `data/fetch_real_smishing.py` and the attribution note in
that file) — 33,869 English-language, already-anonymized real smishing
reports, three of whose categories map directly onto elder-fraud vectors:
`hey mum/dad` (the real "Hi Mum, I broke my phone" family-impersonation
scam — literally our grandparent_scam category), `government` (real
IRS/NHS-style impersonation), and `delivery` (a category we didn't have at
all: fake package/toll delivery-fee smishing, added as a new
`package_delivery_scam` category). After this plus more hand-authored
template variety: grandparent_scam 93.5%, romance_scam 100%,
government_impersonation 98.3%, package_delivery_scam 94.0%.

**4. Feature engineering.** `compare_models.py` tried five configurations on
the validation set: plain word TF-IDF (baseline), word+char n-grams, word +
hand-picked structural features (message length, dollar-sign density,
exclamation density, ALL-CAPS ratio, URL presence), a calibrated Linear SVM,
and word+char+structural combined. The combined word+char+structural
Logistic Regression won (97.15% val accuracy vs. 95.97% baseline) and is
what `train_model.py` now uses.

**A word-boundary bug, found along the way**: the tactic/category keyword
matcher used plain substring matching, so the keyword `"irs"` (for the IRS)
matched inside the word "**f-irs-t**", falsely flagging any message that
used the word "first". Fixed with regex word-boundary matching
(`_find_keyword_hits` in `scam_detector.py`).

**Known remaining limitation**: `legitimate_formal` messages (the exact
category added to fix bug #1) still score elevated on the raw model
(~0.5-0.67, "medium" tier) rather than confidently "safe" — 15 templates
isn't quite enough for a linear bag-of-words model to learn "formal register
is fine *unless* combined with a payment demand" as a general concept,
since formal language itself overlaps with real phishing's vocabulary. The
architectural fix from #1 keeps this contained (none of these ever reach a
false "SCAM" alarm — verified directly), but the most promising next step
is more `legitimate_formal` template variety, and/or a model that can
represent word *interactions* better than a linear one (a small
embedding-based classifier would be the natural next experiment, though
that goes beyond the Stanford/DeepLearning.AI course's linear/logistic
regression scope this project is built on).

If you retrain later (e.g. after expanding the dataset further), run
`data/rebuild_dataset.py` (optionally after `data/fetch_real_smishing.py` to
refresh the real-data pull) then `python model/train_model.py` — `app.py`
loads whatever `.pkl` files currently sit in `model/`, so nothing else needs
to change. Use `model/error_analysis.py` to see exactly where the current
model is weakest before deciding what data to add next.

## Benchmarking against an industry-standard dataset

Deep-learning/LLM papers that get cited as "benchmarks" for scam-text
classification almost never test on the same task or data as this project
(see "ML benchmark research" below), so their numbers aren't a fair
comparison. What *is* a fair, widely-recognized comparison point is the
**SMS Spam Collection dataset** (Almeida & Hidalgo, 2011, UCI Machine
Learning Repository, CC BY 4.0) — the de facto standard academic benchmark
for SMS spam classification, cited in hundreds of papers.

`model/benchmark_industry_standard.py` runs two clearly-separated checks
against `data/industry_benchmark/sms_spam_collection.csv` (5,169 unique
messages, 4,516 ham / 653 spam):

1. **Our technique, trained fresh on this dataset** (fair, apples-to-apples
   comparison): word + char TF-IDF + structural features + Logistic
   Regression — the same pipeline as `train_model.py` — scores **97.87%
   accuracy / 0.9147 F1** on its own held-out test split. That lands right
   in the literature's classic-ML range (Naive Bayes ~95-98%, SVM/Logistic
   Regression + TF-IDF ~97-99%), and close behind the deep-learning numbers
   sometimes cited for this space (ResNet 99.08%/0.9646 F1, CNN-GRU
   98.97%/0.9596 F1 — though those papers used a different, newer Kaggle
   upload of an SMS spam dataset, not confirmed to be this exact corpus, so
   treat that comparison as approximate).
2. **Our actual production elder-scam model, zero-shot on this dataset, no
   retraining** (informational only — a different domain/task, not a fair
   apples-to-apples number): **98.59% accuracy / 0.9425 F1** on the full
   dataset. Interesting on its own terms — a model trained only on
   narrative elder-fraud messages still generalizes well to generic SMS
   spam — but this is a bonus finding, not the headline claim.

Run it yourself with:

```bash
python model/benchmark_industry_standard.py
```

**Takeaway for the writeup**: our word+char+structural TF-IDF + Logistic
Regression technique — the same linear modeling approach from the Stanford/
DeepLearning.AI course this project is built on — is competitive with
classic ML baselines on the exact public dataset the literature uses, and
within a couple of points of published deep-learning results, without the
architecture complexity, training cost, or "black box" trade-off. That's a
legitimate, defensible benchmarking claim; claiming to have "beaten" the
deep-learning papers, or that they're directly comparable to our
elder-fraud-specific model, would not be.

### ML benchmark research (why not deep learning / LLM papers)

Before landing on the SMS Spam Collection dataset above, several
"benchmark" papers surfaced during research (via Gemini, then independently
verified via web search) were checked for whether they're a fair comparison
for this project. All four are real, published papers — but none of them
test the same task:

- **PsyScam** (arXiv 2505.15017) tests whether *LLMs* can recognize/generate
  psychological manipulation techniques in scam text — not a classification-
  accuracy benchmark.
- **FraudSMSWalker** (arXiv 2606.16659) benchmarks *agentic* LLMs following
  masked SMS-to-webpage link chains — a multi-step, tool-using task, not
  single-message classification.
- An XGBoost+TF-IDF paper (arXiv 2604.11752) reports 72.5% accuracy / 0.691
  macro F1 — but on multi-turn conversations (3,201 full exchanges), a
  harder task than classifying one message at a time; its dataset's public
  availability was also unconfirmed as of the paper's writing.
- A ResNet/CNN-GRU paper (Journal of Computer Networks, Architecture and
  High Performance Computing) reports 99.08%/98.97% accuracy on a Kaggle SMS
  spam dataset — the closest structural match (single-message binary
  classification), and the basis for the "literature context" numbers cited
  above, but on generic spam, not elder-fraud-specific text.

Conclusion: none of these are a fair head-to-head benchmark for this
project's actual task, and switching architectures to chase them would be a
scope/authenticity mismatch given this project's goal of showcasing the
Stanford/DeepLearning.AI logistic-regression coursework. The SMS Spam
Collection comparison above is the honest, defensible substitute.

## Optional: comparing against a frontier LLM

The shipped app never calls an LLM — it's a fully offline, local model by
design (see rationale above). But "would a frontier LLM actually do
better?" is a fair question, and one worth having a real, measured answer
to rather than a guess. `model/benchmark_llm_vs_baseline.py` runs the exact
same held-out test messages through both our local model and Claude Haiku
(via the Anthropic API), and reports accuracy, F1, per-message latency, and
dollar cost for each, side by side.

This is a separate, optional tool — not part of the app or its dependencies:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...     # your own key, from console.anthropic.com
python model/benchmark_llm_vs_baseline.py               # 150-message sample, asks to confirm cost
python model/benchmark_llm_vs_baseline.py --full --yes  # entire test set (~1,018 rows, ~$1, ~10-15 min)
```

It prints an estimated cost before spending anything (Claude Haiku is
cheap — a 150-message sample runs under $0.25) and an exact cost afterward
from real token usage. Use `--csv results.csv` to save every message's
verdict from both models (including the LLM's plain-language reasoning)
for closer inspection or for figures in the writeup.

Whatever the numbers come out to, the honest framing for the writeup is:
this measures whether a general-purpose frontier model can match a small,
purpose-built classifier on this exact task — not whether one is
"smarter" in general. A well-tuned local model beating or matching an LLM
here would say more about how well-suited a lightweight classifier is to
short, formulaic scam text than about model capability in the abstract.

## Running locally

```bash
pip install -r requirements.txt
python model/train_model.py      # regenerates the .pkl files for YOUR scikit-learn version
python app.py
```

Then open http://localhost:5000.

**Always run `train_model.py` after a fresh `pip install` on a new machine.**
The `.pkl` files are tied to the exact scikit-learn version that created
them — if yours differs, you'll get an `InconsistentVersionWarning` (usually
harmless, but not worth risking right before a demo). Regenerating them
locally with `data/elder_focus_train/val/test.csv` (already included) takes
a few seconds and reproduces the same ~97.5% test accuracy.

To rebuild the dataset itself from scratch (e.g. after adding more examples
to `generate_elder_examples.py`, or to refresh the real-smishing pull):

```bash
python data/fetch_real_smishing.py   # optional -- re-downloads real smishing data (needs internet)
python data/generate_elder_examples.py
python data/rebuild_dataset.py       # merges everything, does the leakage-safe grouped split
python model/train_model.py
python model/error_analysis.py       # see where the new model is weakest
```

Note: `data/rebuild_dataset.py` needs `elder_scam_dataset_full.csv` (the
original 40k-row full corpus) in `data/` -- it's not included in this zip to
keep the download size reasonable. Get it from the original
`elder_scam_dataset.zip` and drop `elder_scam_dataset_full.csv` into `data/`
if you want to rebuild from scratch; the current `elder_focus_train/val/
test.csv` files already reflect the latest rebuild, so this is only needed
if you're changing the source data.

### Troubleshooting

- **`Address already in use` / port 5000 taken**: on macOS, AirPlay Receiver
  listens on port 5000 by default. Either turn it off in System Settings →
  General → AirDrop & Handoff, or run on a different port:
  `PORT=5001 python3 app.py`, then open http://localhost:5001.
- **`InconsistentVersionWarning` from scikit-learn**: run
  `python model/train_model.py` to regenerate the `.pkl` files with your
  installed scikit-learn version (see above).

## How the risk assessment works

- **Model layer**: TF-IDF + Logistic Regression (`baseline_classifier.py`'s
  approach, matching the classification techniques from the Stanford/
  DeepLearning.AI Supervised Machine Learning course), extended with
  character n-grams (catches spelling tricks like "arnaz0n") and a few
  structural features (message length, dollar-sign density, exclamation
  density, ALL-CAPS ratio, URL presence) after `compare_models.py` showed
  this combination generalizes better than word TF-IDF alone. Still
  fundamentally the same linear classifier from the course, just with a
  richer feature set (`model/feature_utils.py` builds it consistently for
  both training and inference).
- **Explanation layer** (`model/scam_detector.py`):
  - Pulls the actual n-grams from *this* message that most drove the "scam"
    score (TF-IDF weight × logistic regression coefficient — a standard
    linear-model explainability trick).
  - Matches the message against a curated list of well-known scam tactics
    (urgency, secrecy, unusual payment methods, authority impersonation,
    threats, too-good-to-be-true offers, personal info requests, remote
    access requests) — shown to the user as "Why we flagged this."
  - Matches against the 8 IC3-grounded scam categories from the dataset
    (grandparent scam, government impersonation, tech support scam, romance
    scam, investment fraud, gift-card/BEC scam, lottery scam, phishing) to
    give a specific "This matches a Grandparent Scam" label plus tailored
    "what to do" advice.
  - **Sender check (optional)**: if the user provides the sender's email
    address or phone number, checks the email domain against a small starter
    list of known official domains (IRS, SSA, Medicare, major banks/couriers/
    tech companies — see `KNOWN_OFFICIAL_DOMAINS` in `scam_detector.py`, easy
    to extend with your own doctor/pharmacy/bank). A matching domain is
    reassuring evidence; a free email address (gmail.com, etc.) or non-`.gov`
    domain claiming to be a government agency is a red flag. This is
    string-matching on the address only, not real sender authentication
    (SPF/DKIM) — a determined attacker can still spoof a "From" address, so
    it's a heuristic aid, not a guarantee, and is shown to users as such.
- A "high risk / SCAM" verdict requires at least one concrete red flag
  (a matched tactic, category, or suspicious sender signal) — not just a raw
  model score with nothing to point to (see the false-positive fix above).
  A verified matching sender domain, with no other red flags found, pulls a
  borderline score down to "looks safe." Final tiers: **high** (≥0.70 *and*
  at least one red flag) "This looks like a SCAM", **medium** (0.35–0.70, or
  ≥0.70 with no red flag found) "Be careful" / "We're not fully sure",
  **low** (<0.35, or reassured by a verified sender) "This looks safe."

## Senior-friendly UI choices

- Large base font size (19px+) with an on-page A+/A− text-size control
  (remembered across visits).
- High-contrast traffic-light result cards that use color **and** shape
  **and** an icon **and** a plain-language label — never color alone. The
  three risk icons borrow their shapes from universal road signs (circle =
  safe/go, triangle = caution, octagon = stop/danger), so the verdict reads
  even for colorblind users or at a glance.
- No ML jargon surfaced by default (no "confidence score" front and center);
  explanations are everyday language.
- Big touch targets, minimal navigation, a single clear call to action per
  screen.
- Works with JavaScript disabled (plain form POST, server-rendered result) —
  `static/js/app.js` progressively enhances it with a no-reload fetch submit,
  example messages, and an optional "Read this aloud" button
  (`speechSynthesis`, no external dependency).

### Visual design pass (2026-08-05)

The app went from "functionally senior-friendly" to an intentional design
pass, on the theory that Design is one of three equally-weighted CAC judging
categories (alongside Concept and Skill) and deserved the same attention as
the model itself:

- **Typeface**: body and headings use [Atkinson Hyperlegible](https://brailleinstitute.org/freefont),
  a typeface designed by the Braille Institute specifically for readers with
  low vision — a deliberate choice for this exact audience, not just a
  nicer-looking font. Self-hosted as local `.woff2` files in `static/fonts/`
  (SIL Open Font License, `static/fonts/OFL.txt`) rather than loaded from a
  CDN, so the app keeps working with zero internet connection, consistent
  with its offline-by-design philosophy.
- **Icons**: hand-drawn inline SVGs (`templates/_icons.html`, mirrored in
  `static/js/app.js` for the no-reload path) replace the emoji used earlier.
  Emoji render inconsistently across operating systems (a real risk when
  demoing on unfamiliar hardware) — SVGs render identically everywhere and
  can be colored with CSS to match the verdict.
- **Favicon/branding**: a shield-checkmark mark used consistently as the
  header logo, browser tab favicon (`static/favicon.ico`, multi-resolution),
  and home-screen icon (`static/img/apple-touch-icon.png`) — generated from
  a single source SVG (`static/img/favicon-source.svg`) so they never drift
  out of sync.
- **Layout/spacing**: moved from ad-hoc spacing values to a consistent 4px
  spacing scale, refined card shadows/depth, a colored left-edge accent on
  result cards (in addition to the background tint) for one more layer of
  at-a-glance scannability, and subtle hover/focus states throughout.
- **Verified, not assumed**: rendered the app with Playwright (screenshots
  of the home page, all three risk tiers, the About page, the no-JS
  fallback path, and the text-size control at max zoom) to confirm the
  redesign actually works end to end rather than trusting the CSS alone.
  Also computed WCAG contrast ratios for every text/background pairing in
  the palette — all meet at least AA, most meet AAA (7:1+), appropriate
  for an audience that may include low-vision users.

## Suggested next steps

1. Keep testing with real-world messages you receive yourself (this is how
   the Sharp Healthcare bug was found) — run them through `/check`, and if
   something looks wrong, add it as a new test case and re-run
   `model/error_analysis.py` to see if it's a one-off or part of a pattern.
2. Add more `legitimate_formal` template variety (see the known limitation
   above) — this is the most likely lever left to fully close that gap.
3. Consider a favicon / app icon and a short demo script for judges (there's
   already an `/about` page summarizing how it works and current metrics).
4. If time allows: expand the tactic/category dictionaries in
   `scam_detector.py` with more phrasing as new test cases surface gaps.
5. `data/fetch_real_smishing.py` only pulled from one real-world source
   (reportsmishing/Smishing-Dataset-IMC25). If more real elder-scam-relevant
   data is needed later, check licensing carefully first -- a couple of
   other candidates turned up during this search
   (shaghayegh-hp/Smishing_Dataset on GitHub, a Mendeley SMS phishing
   dataset) but had ambiguous or unstated licenses, so they weren't used
   here. Worth a direct email to the authors if you want to use them.
