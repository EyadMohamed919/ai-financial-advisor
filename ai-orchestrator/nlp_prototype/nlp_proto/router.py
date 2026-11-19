"""Message router: decides what a chat message is for, before any extraction happens.

Routes (matching the team diagram: Classifier / Forecast / APIs + LLM):
  transaction    -> transaction extraction pipeline (FR-010..FR-013)
  spending_query -> computed from the user's own records (FR-018)
  forecast       -> spending projection from the user's records
  market         -> economic/market data APIs + LLM explanation (FR-023)
  advice         -> LLM guidance with a minimal data summary (FR-022, FR-029)
  manage         -> undo/delete on recorded entries (FR-016)
  greeting, out_of_scope -> fixed replies

Usage: python -m nlp_proto.router   trains, evaluates on a held-out 20%, saves models/router.pkl
"""
import collections
import glob
import json
import os
import pickle
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import DATA, MODELS, ROOT, TX_INTENTS, load_jsonl, load_synthetic
from .normalize import normalize

ROUTES = ["transaction", "spending_query", "forecast", "market", "advice", "manage", "greeting", "out_of_scope"]
ROUTE_AR = {"transaction": "تسجيل معاملة", "spending_query": "سؤال عن مصاريفي", "forecast": "توقع",
            "market": "أسواق واقتصاد", "advice": "نصيحة", "manage": "تعديل/مسح", "greeting": "تحية",
            "out_of_scope": "خارج النطاق"}
MAX_TRANSACTION_EXAMPLES = 500  # keep the 1,700+ transaction messages from swamping the other routes
# Character n-grams see "هصرف" (will spend) and "صرفت" (spent) as near-identical, so a question in the
# Egyptian future tense that the model calls spending_query is a forecast.
FUTURE_RE = re.compile(
    r"(?:^|\s)(?:ه|هي|هن)(?:صرف|حوش|وفر|دفع|فضل|تبقي|كفي|خلص|قبض|يفضل|يتبقي|يكفي|يخلص)\w*"
    r"|\bwill i\b|\bam i going to\b|\bhow much will\b|\bha(?:sraf|7awesh|dfa3)\b|\bhayfdal\b", re.I)


def load_router_data(seed=42):
    """Combine new router data, relabelled questions/corrections, and a sample of transaction messages."""
    synthetic = load_synthetic()
    by_id = {r["id"]: r for r in synthetic}
    items, relabels = [], {}
    for path in sorted(glob.glob(os.path.join(DATA, "router", "*.jsonl"))):
        for r in load_jsonl(path):
            if "relabel_of" in r:
                relabels[r["relabel_of"]] = r["route"]
            else:
                items.append((r["text"], r["route"]))
    for rid, route in relabels.items():
        items.append((by_id[rid]["text"], route))
    items += [(r["text"], "greeting") for r in synthetic if r["intent"] == "greeting"]
    tx = [r["text"] for r in synthetic if r["intent"] in TX_INTENTS]
    random.Random(seed).shuffle(tx)
    items += [(t, "transaction") for t in tx[:MAX_TRANSACTION_EXAMPLES]]
    return items


def split(items, test_ratio=0.2, seed=42):
    groups = collections.defaultdict(list)
    for text, route in items:
        groups[route].append((text, route))
    rng = random.Random(seed)
    train, test = [], []
    for route in sorted(groups):
        g = sorted(groups[route])
        rng.shuffle(g)
        n = max(1, round(len(g) * test_ratio))
        test += g[:n]
        train += g[n:]
    return train, test


class Router:
    path = os.path.join(MODELS, "router.pkl")
    unsure_below = 0.35  # below this the router asks what the user meant

    def train(self, items, save=True):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            LogisticRegression(max_iter=3000, C=10, class_weight="balanced"),
        ).fit([normalize(t) for t, _ in items], [r for _, r in items])
        if save:
            os.makedirs(MODELS, exist_ok=True)
            with open(self.path, "wb") as fh:
                pickle.dump(self.model, fh)
        return self

    def load(self):
        with open(self.path, "rb") as fh:
            self.model = pickle.load(fh)
        return self

    def predict(self, text):
        probs = self.model.predict_proba([normalize(text)])[0]
        ranked = sorted(zip(probs, self.model.classes_), reverse=True)
        (p, route), (p2, route2) = ranked[0], ranked[1]
        route = str(route)
        if route == "spending_query" and FUTURE_RE.search(normalize(text)):
            route = "forecast"
        return {"route": route, "confidence": float(p), "runner_up": str(route2), "unsure": bool(p < self.unsure_below)}


def main():
    items = load_router_data()
    train, test = split(items)
    router = Router().train(train, save=False)
    preds = [router.predict(t)["route"] for t, _ in test]
    gold = [r for _, r in test]
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    per_route = {r: (sum(p == g == r for p, g in zip(preds, gold)), gold.count(r)) for r in ROUTES if gold.count(r)}
    confusions = collections.Counter((g, p) for p, g in zip(preds, gold) if p != g).most_common(10)
    errors = [(t, g, p) for (t, g), p in zip(test, preds) if p != g][:25]
    lines = [f"Router: trained on {len(train)}, tested on {len(test)} held-out messages. Accuracy **{acc:.1%}**", "",
             "| Route | Correct / total | Accuracy |", "|---|---|---|"]
    lines += [f"| {r} ({ROUTE_AR[r]}) | {c} / {n} | {c / n:.1%} |" for r, (c, n) in per_route.items()]
    lines += ["", "Most common confusions (gold → predicted): " + ", ".join(f"{g} → {p} ({n})" for (g, p), n in confusions), "",
              "Sample errors:", ""] + [f"- `{t}`: gold {g}, predicted {p}" for t, g, p in errors]
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "router_results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    Router().train(items)  # final model on all data, used by the chat
    print(f"saved {Router.path} (trained on all {len(items)} messages)")


if __name__ == "__main__":
    main()
