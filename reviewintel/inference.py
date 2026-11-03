"""inference.py - load the trained models and score one review (NumPy only; no sklearn / pandas needed)."""
import json
import os
import re

import numpy as np

from .features import FeatureExtractor, URL_RE, PHONE_RE, hand_crafted, HAND_NAMES
from .nn import MLP
from .topics import detect_topics


class Predictor:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.fx = FeatureExtractor.load(os.path.join(model_dir, "features.json"))
        self.spam = MLP.load(os.path.join(model_dir, "spam_model.npz"))
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.thr = float(self.meta.get("threshold", 0.5))
        sp = os.path.join(model_dir, "sentiment_model.npz")
        self.sent = MLP.load(sp) if os.path.exists(sp) else None
        self.sent_names = self.meta.get("sentiment_classes", ["Negative", "Neutral", "Positive"])

    # -------------------------------------------------------------- helpers
    def _spam_prob(self, texts):
        return self.spam.predict_proba(self.fx.transform(texts))[:, 0]

    def _signals(self, text):
        h = dict(zip(HAND_NAMES, hand_crafted(text)))
        sig = []
        if h["has_url"]:
            sig.append("Contains a web link")
        if h["has_phone"]:
            sig.append("Contains a phone number")
        if re.search(r"@\w+", text):
            sig.append("Contains a social handle (@...)")
        if text.count("!") >= 3:
            sig.append("Excessive exclamation marks")
        if h["caps_ratio"] > 0.6 and len(text) > 15:
            sig.append("Mostly CAPITAL letters")
        if h["repeat_ratio"] > 0.35:
            sig.append("Many repeated words")
        if len(text.split()) < 3:
            sig.append("Very short message")
        return sig

    def _influence(self, text, base, top=5):
        """Occlusion: how much does the spam probability drop if a word is removed?"""
        words = text.split()[:60]
        if len(words) < 2:
            return []
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        delta = base - self._spam_prob(variants)
        order = np.argsort(-np.abs(delta))[:top]
        return [dict(word=words[i], effect=round(float(delta[i]), 3)) for i in order if abs(delta[i]) > 0.01]

    # -------------------------------------------------------------- main API
    def predict(self, text):
        text = (text or "").strip()
        if not text:
            raise ValueError("Please enter a review message.")
        text = text[:2000]
        p = float(self._spam_prob([text])[0])
        is_spam = p >= self.thr
        out = dict(label="Spam" if is_spam else "Not Spam", is_spam=bool(is_spam), spam_probability=round(p, 4),
                   confidence=round(p if is_spam else 1 - p, 4), threshold=self.thr,
                   topics=[] if is_spam else detect_topics(text), signals=self._signals(text), influential_words=self._influence(text, p),
                   sentiment=None)
        if self.sent is not None and not is_spam:
            sp = self.sent.predict_proba(self.fx.transform([text]))[0]
            out["sentiment"] = dict(label=self.sent_names[int(sp.argmax())],
                                    probs={n: round(float(v), 4) for n, v in zip(self.sent_names, sp)})
        return out
