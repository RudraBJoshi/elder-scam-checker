# Accuracy & Reliability Upgrades — Sep 20-21, 2026

Summary of everything changed in this pass, across two commits
(`b1b3791`, `8e66aba`) plus follow-up fixes made while testing against
real messages. Starting point: 97.5% test accuracy, but weak in specific,
important ways (formal legitimate notices, links, and vague verdicts).

## 1. Training data (`data/generate_extra_examples.py`, new file)

Added ~45 new templates, split by template into train/val/test so no
wording crosses the boundary (same leakage-safe rule as the original
`rebuild_dataset.py`):

- **Legitimate notices the model had never seen**: bank/card alerts,
  appointment and pharmacy reminders, one-time codes, bills, delivery
  confirmations, family texts, and — added after a real false positive —
  ordinary account/security emails ("you're receiving this because you
  signed in with Google", password-changed notices, receipts).
- **Harmless links**: photo albums, recipes, Zoom invites, official sites
  (previously *any* link pushed the score to 0.9+ as a scam).
- **Scam styles the dataset barely covered**: fake refund/subscription
  callback numbers, military romance requests, gift-card favor asks,
  Medicare/benefit phishing, lookalike-domain smishing.
- Formal-legitimate rows are upweighted 8x in `train_model.py` — cut
  their validation false-alarm rate from 22/40 to 5/40 with no new
  missed scams.

## 2. Link & sender analysis (`model/scam_detector.py`)

New, previously nonexistent capability:

- Flags lookalike domains (`amaz0n-secure-login.top`, `medicare.gov.evil.top`),
  shortened links (bit.ly, tinyurl, etc.), and raw-IP links.
- Extended the official-domain and brand-impersonation lists (Netflix,
  Walmart, Google, banks, carriers, etc.).
- A deceptive link or spoofed sender domain is now concrete evidence on
  its own — it floors the verdict at "Be careful" even if the message
  wording alone looks bland, and a sender impersonating a government
  agency from a free email address goes straight to "high" risk.

## 3. Rule-layer detection gaps closed

Added/broadened tactic rules:

- **Emergency/kidnapping money demands** ("help me, I've been kidnapped,
  send $10,000") — missed entirely before.
- **Generic money requests** ("can I have $50,000") — previously only
  caught specific phrasings tied to other categories.
- **"Fix your account" pretext**, including misspellings ("acocunt") and
  texting shorthand ("ur", "u").
- **Callback-number scams** ("call us about this charge/refund/virus").
- **Utility shut-off threats**, **"earn $X/day"** work-from-home bait,
  more gift-card and secrecy phrasing ("keep it private", "tell no one").
- Broadened company-impersonation keywords (Google, Facebook, WhatsApp,
  Netflix — a real "Hello this is google!" scam was missed before this).

## 4. Verdict logic rework

- A bare brand name or a single loose keyword no longer counts as
  evidence by itself.
- **Escalation now depends on rule evidence, not just the raw model
  score**: two strong red flags, or one strong flag plus a money
  request, escalates to "high" regardless of model probability. This
  closed a real gap where friendly filler text ("Hope you're well!")
  diluted the model score enough to hide an otherwise obvious scam.
- Renamed the vague "We're not fully sure — worth a second look" to
  "This looks suspicious — check before you act" for a clearer,
  less wishy-washy headline.

## 5. Result content rework (backend + both renderers)

`analyze()` now always returns:
- `reasons` — plain-language bullets, always non-empty on any non-safe
  verdict (previously could say "we didn't find warning signs" while
  still flagging the message — the exact complaint that started this).
- `advice` — category-specific guidance, or generic advice when no
  specific category matched.
- `next_steps` — a 3-item universal checklist (don't reply/click/pay,
  call using a number you already have, tell someone you trust).
- `reassurance` — why a "safe" verdict is safe, when applicable.

Updated both the Jinja template (`templates/_result.html`) and the
client-side JS renderer (`static/js/app.js`, including the "Read This
Aloud" text) to use these fields consistently, and added CSS for the new
checklist.

## 6. Reliability fixes

- **Crash fix**: `{"message": null}` or non-string `message`/`sender`
  values 500'd the `/check` route. Fixed in `app.py`.
- Whitespace normalization so double-spaced or newline-heavy text
  doesn't dodge keyword rules.
- Verified no ReDoS risk in the new regex-based rules under adversarial
  input (long repeated strings, huge dotted hostnames).

## 7. Testing added (`tests/`, new)

Three standalone scripts (no pytest dependency):
- `run_battery.py` — 30 fresh-wording scams + 30 fresh-wording
  legitimate messages, plus 4 sender-domain cases.
- `run_robustness.py` — malformed/empty input, wrong JSON types, 100k-char
  messages, XSS payloads, odd sender values, ReDoS probes, determinism.
- `run_mutations.py` — every case run through 8 mutations (ALL CAPS,
  typos, emoji, filler text, spacing/newline changes) to check how much
  evasion the rules tolerate.

## Results

| Check | Before | After |
|---|---|---|
| Test accuracy | 97.5% | 98.1% |
| Validation accuracy | 97.4% | 98.0% |
| Held-out formal legit notices correct | 21% | ~90% |
| 26 fresh unseen messages (manual probe) | 62% | 85-92%* |
| Fresh 30-scam / 30-legit battery | not tested | 30/30 and 30/30 |
| Message with a harmless link | scored 0.9+ as scam | scores 0.03-0.06 |

\* range depends on which templates land in the held-out split; treat as
directional, not a benchmark (only 26 messages).

## Known remaining gaps

- Romance/favor scams with no trigger words can still score low.
- Some real transaction/utility alerts still land in "worth a second look" territory.
- ALL-CAPS versions of ~2/30 legitimate messages get flagged (caps ratio
  is a genuine scam signal in the training data, so this is a tradeoff,
  not an oversight).
- Scams buried in enough friendly filler text (~4/30 in mutation testing)
  can still slip under "low" when they only trip weak rules.

## Commits

- `b1b3791` — data, link/sender analysis, verdict rework, README update.
- `8e66aba` — input-crash fix, escalation-logic tightening, rule
  additions, test suite.

Both are pushed to `origin/main`. (Render auto-deploy was separately
found to be broken due to a GitHub App permissions issue — Render
reported "It looks like we don't have access to your repo" — unrelated
to app code; needs to be fixed in Render's GitHub App settings.)
