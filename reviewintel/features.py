r"""
features.py - Text pre-processing and feature extraction (NumPy only).

Pipeline
    raw review  ->  clean_text()  ->  TF-IDF (uni+bi-grams, top-K vocabulary)      [semantic features]
                \->  hand-crafted()   ->  10 numeric 'spam-signal' features          [style features]
    final vector = [ TF-IDF (K) | standardised hand-crafted (10) ]

The extractor is serialised to JSON (not pickle) so the web app needs only NumPy + Flask.
"""
import json
import math
import re
from collections import Counter

import numpy as np

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:in|com|co\.in|net|org)\S*|\bbit\.ly/\S+|\bwa\.me/\S+)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\-\s]?)?[6-9]\d{9}(?!\d)")
HANDLE_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z0-9']+")

HAND_NAMES = ["log_chars", "log_words", "caps_ratio", "digit_ratio", "log_exclaims",
              "has_url", "has_phone", "repeat_ratio", "special_ratio", "avg_word_len"]


def clean_text(text):
    """lower-case, keep URL / phone / handle as tokens, drop emoji + punctuation, squeeze elongations."""
    t = str(text)
    t = URL_RE.sub(" urltoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = HANDLE_RE.sub(" handletoken ", t)
    t = t.lower()
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)            # sooooo -> soo
    t = re.sub(r"[^a-z0-9' ]+", " ", t)              # strips emoji / punctuation
    return re.sub(r"\s+", " ", t).strip()


def tokens(clean):
    return TOKEN_RE.findall(clean)


def ngrams(toks):
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


def hand_crafted(raw):
    """10 numeric features that capture *how* a review is written (not only what it says)."""
    s = str(raw)
    n = max(len(s), 1)
    words = s.split()
    nw = max(len(words), 1)
    letters = [c for c in s if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    digits = sum(c.isdigit() for c in s) / n
    low = [w.lower().strip(".,!?") for w in words]
    repeat = 1.0 - len(set(low)) / nw
    special = sum((not c.isalnum()) and (not c.isspace()) for c in s) / n
    return [math.log1p(len(s)), math.log1p(nw), caps, digits, math.log1p(s.count("!")),
            1.0 if URL_RE.search(s) else 0.0, 1.0 if PHONE_RE.search(s) else 0.0,
            repeat, special, float(np.mean([len(w) for w in words])) if words else 0.0]


class FeatureExtractor:
    def __init__(self, max_features=1500, min_df=3):
        self.max_features, self.min_df = max_features, min_df
        self.vocab, self.idf = {}, None
        self.mu, self.sd = None, None

    # ---------------------------------------------------------------- fit
    def fit(self, texts):
        df, tf = Counter(), Counter()
        for t in texts:
            grams = ngrams(tokens(clean_text(t)))
            tf.update(grams)
            df.update(set(grams))
        cand = [g for g, c in df.items() if c >= self.min_df]
        cand.sort(key=lambda g: (-tf[g], g))
        cand = cand[: self.max_features]
        self.vocab = {g: i for i, g in enumerate(cand)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[g])) + 1.0 for g in cand], dtype=np.float32)
        H = np.array([hand_crafted(t) for t in texts], dtype=np.float64)
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, texts):
        texts = list(texts)
        V = len(self.vocab)
        X = np.zeros((len(texts), V + len(HAND_NAMES)), dtype=np.float32)
        for r, t in enumerate(texts):
            cnt = Counter(g for g in ngrams(tokens(clean_text(t))) if g in self.vocab)
            for g, c in cnt.items():
                j = self.vocab[g]
                X[r, j] = (1.0 + math.log(c)) * self.idf[j]       # sub-linear tf * idf
            nrm = np.linalg.norm(X[r, :V])
            if nrm > 0:
                X[r, :V] /= nrm                                   # L2 normalise
            X[r, V:] = (np.array(hand_crafted(t)) - self.mu) / self.sd
        return X

    @property
    def n_features(self):
        return len(self.vocab) + len(HAND_NAMES)

    # ---------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(max_features=self.max_features, min_df=self.min_df, vocab=list(self.vocab),
                           idf=self.idf.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist()), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        o = cls(d["max_features"], d["min_df"])
        o.vocab = {g: i for i, g in enumerate(d["vocab"])}
        o.idf = np.array(d["idf"], dtype=np.float32)
        o.mu, o.sd = np.array(d["mu"]), np.array(d["sd"])
        return o
