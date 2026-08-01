"""
Shared feature-building code for the scam classifier, used by both
train_model.py (fitting) and scam_detector.py (inference) so the two never
drift out of sync.

Feature set (chosen via compare_models.py, which tried several
configurations and picked this one on validation-set accuracy/F1):
  - word TF-IDF (1,2-grams)      -- the original baseline signal
  - char TF-IDF (3-5 char n-grams, word-boundary-aware) -- catches spelling
    tricks and obfuscation a word-level model misses (e.g. "arnaz0n",
    "paypa1", stylized/leetspeak scam text)
  - structural features           -- message length, dollar-sign density,
    exclamation density, ALL-CAPS word ratio, URL-like token presence --
    signals a bag-of-words model can't represent well on its own
"""
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack


def structural_features(texts):
    rows = []
    for t in texts:
        t = str(t)
        length = len(t)
        words = t.split()
        n_words = max(len(words), 1)
        n_dollar = t.count("$")
        n_exclaim = t.count("!")
        allcaps_ratio = sum(1 for w in words if len(w) > 2 and w.isupper()) / n_words
        has_url = 1.0 if re.search(r"https?://|www\.|bit\.ly", t, re.I) else 0.0
        rows.append([
            length / 500.0,
            n_dollar / n_words,
            n_exclaim / n_words,
            allcaps_ratio,
            has_url,
        ])
    return csr_matrix(np.array(rows, dtype=float))


def build_features(word_vec, char_vec, texts):
    """Combine word TF-IDF + char TF-IDF + structural features into one
    sparse matrix, in a fixed column order (word | char | structural) that
    both training and inference must agree on."""
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    Xs = structural_features(texts)
    return hstack([Xw, Xc, Xs]).tocsr()
