"""Approach B: character n-gram TF-IDF + logistic regression (intent and category) + rule-based amounts."""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .amounts import extract_amount
from .data import MODELS, TX_INTENTS
from .normalize import normalize


def _model():
    # Character n-grams cope with Arabic prefixes, typos and Franco-Arabic without a tokenizer.
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=3000, C=10),
    )


class TfidfApproach:
    name = "B. TF-IDF + Logistic Regression"
    path = os.path.join(MODELS, "tfidf.pkl")

    def train(self, recs):
        self.intent = _model().fit([normalize(r["text"]) for r in recs], [r["intent"] for r in recs])
        tx = [r for r in recs if r["intent"] in TX_INTENTS and r["category"]]
        # Prefix the intent so "Other" income and "Other" expense stay distinct.
        self.category = _model().fit([normalize(r["text"]) for r in tx], [f"{r['intent']}|{r['category']}" for r in tx])
        os.makedirs(MODELS, exist_ok=True)
        with open(self.path, "wb") as fh:
            pickle.dump((self.intent, self.category), fh)

    def load(self):
        with open(self.path, "rb") as fh:
            self.intent, self.category = pickle.load(fh)
        return self

    def size_bytes(self):
        return os.path.getsize(self.path)

    def predict(self, text):
        t = normalize(text)
        intent = self.intent.predict([t])[0]
        category, conf = None, None
        if intent in TX_INTENTS:
            probs = self.category.predict_proba([t])[0]
            classes = self.category.classes_
            # Only categories valid for the predicted intent.
            allowed = [(p, c) for p, c in zip(probs, classes) if c.startswith(intent + "|")]
            if allowed:
                p, c = max(allowed)
                category, conf = c.split("|", 1)[1], p / sum(p2 for p2, _ in allowed)
        return {"intent": intent, "category": category, "category_conf": conf, **extract_amount(text), "entities": []}
