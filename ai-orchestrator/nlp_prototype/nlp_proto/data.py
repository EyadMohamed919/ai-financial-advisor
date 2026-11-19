"""Dataset loading, the fixed train/test split, and label mapping."""
import glob
import json
import os
import random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

INTENTS = ["log_expense", "log_income", "log_savings", "log_transfer", "question", "greeting", "other"]
TX_INTENTS = {"log_expense", "log_income", "log_savings", "log_transfer"}
# Coarse type used to compare against the hand-written set, which only has a "type" column.
COARSE = {"log_expense": "expense", "log_income": "income", "log_savings": "savings",
          "log_transfer": "transfer", "question": "none", "greeting": "none", "other": "none"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_synthetic():
    recs = []
    for path in sorted(glob.glob(os.path.join(DATA, "train", "*.jsonl"))):
        recs.extend(load_jsonl(path))
    return recs


def split(recs, test_ratio=0.2, seed=42):
    """Deterministic split, stratified by intent so every intent appears in both parts."""
    by_intent = defaultdict(list)
    for r in recs:
        by_intent[r["intent"]].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for intent in sorted(by_intent):
        group = sorted(by_intent[intent], key=lambda r: r["id"])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def load_handwritten():
    """80 hand-written messages (different author and style from the synthetic data)."""
    recs = load_jsonl(os.path.join(DATA, "eval_handwritten.jsonl"))
    for r in recs:
        r["coarse"] = r["type"]
    return recs


def entity_offsets(rec):
    """Convert {'label', 'text'} entities to (start, end, label) character offsets."""
    out = []
    for e in rec["entities"]:
        start = rec["text"].find(e["text"])
        if start >= 0:
            out.append((start, start + len(e["text"]), e["label"]))
    return out
