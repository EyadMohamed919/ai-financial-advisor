"""Train every approach on the same split and compare them.

Usage: python -m nlp_proto.evaluate
Writes results/results.md and results/results.json.
"""
import json
import os
import time

from .data import COARSE, ROOT, TX_INTENTS, entity_offsets, load_handwritten, load_synthetic, split
from .rules import RulesApproach
from .spacy_model import SpacyApproach
from .tfidf import TfidfApproach


def _same_amount(pred, gold_amount, gold_currency):
    return pred["amount"] is not None and abs(pred["amount"] - gold_amount) < 1e-6 and pred["currency"] == gold_currency


def score(preds, gold, has_fine_intent):
    n = len(gold)
    m = {"n": n}
    coarse_pred = [COARSE[p["intent"]] for p in preds]
    coarse_gold = [COARSE[g["intent"]] if has_fine_intent else g["coarse"] for g in gold]
    m["type_acc"] = sum(a == b for a, b in zip(coarse_pred, coarse_gold)) / n
    if has_fine_intent:
        m["intent_acc"] = sum(p["intent"] == g["intent"] for p, g in zip(preds, gold)) / n
    with_amount = [(p, g) for p, g in zip(preds, gold) if g["amount"] is not None]
    m["amount_acc"] = sum(_same_amount(p, g["amount"], g["currency"]) for p, g in with_amount) / len(with_amount)
    with_cat = [(p, g) for p, g in zip(preds, gold) if g["category"] is not None]
    m["category_acc"] = sum(p["category"] == g["category"] for p, g in with_cat) / len(with_cat)
    ok = 0
    for p, g, cp, cg in zip(preds, gold, coarse_pred, coarse_gold):
        if cg == "none":
            ok += cp == "none"
        else:
            ok += (cp == cg and (g["amount"] is None or _same_amount(p, g["amount"], g["currency"]))
                   and (g["category"] is None or p["category"] == g["category"]))
    m["end_to_end"] = ok / n
    return m


def ner_scores(preds, gold):
    """Micro P/R/F1 over (label, text) pairs, excluding nothing."""
    tp = fp = fn = 0
    for p, g in zip(preds, gold):
        pred = {(e["label"], e["text"]) for e in p["entities"]}
        true = {(label, g["text"][s:e]) for s, e, label in entity_offsets(g)}
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0}


def main():
    synthetic = load_synthetic()
    train, test = split(synthetic)
    handwritten = load_handwritten()
    results = {"train_size": len(train), "test_size": len(test), "handwritten_size": len(handwritten), "approaches": []}
    for approach in (RulesApproach(), TfidfApproach(), SpacyApproach()):
        t0 = time.perf_counter()
        approach.train(train)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        test_preds = [approach.predict(r["text"]) for r in test]
        hw_preds = [approach.predict(r["text"]) for r in handwritten]
        latency_ms = (time.perf_counter() - t0) / (len(test) + len(handwritten)) * 1000
        row = {"name": approach.name, "train_seconds": round(train_s, 1), "latency_ms": round(latency_ms, 2),
               "size_mb": round(approach.size_bytes() / 1e6, 2),
               "synthetic_test": score(test_preds, test, True), "handwritten": score(hw_preds, handwritten, False)}
        if isinstance(approach, SpacyApproach):
            row["ner"] = ner_scores(test_preds, test)
            row["misaligned_entities_skipped"] = approach.misaligned
        row["handwritten_errors"] = [
            {"id": g["id"], "text": g["text"], "gold": [g["coarse"], g["amount"], g["category"]],
             "pred": [COARSE[p["intent"]], p["amount"], p["category"]]}
            for p, g in zip(hw_preds, handwritten)
            if not (COARSE[p["intent"]] == g["coarse"] and (g["amount"] is None or _same_amount(p, g["amount"], g["currency"]))
                    and (g["category"] is None or p["category"] == g["category"]))]
        results["approaches"].append(row)
        print(f"done: {approach.name}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    md = render(results)
    with open(os.path.join(ROOT, "results", "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


def render(r):
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [f"Train: {r['train_size']} synthetic · Test: {r['test_size']} synthetic (held out) + {r['handwritten_size']} hand-written", "",
             "| Approach | Set | Type | Intent | Amount | Category | End-to-end |", "|---|---|---|---|---|---|---|"]
    for a in r["approaches"]:
        s, h = a["synthetic_test"], a["handwritten"]
        lines.append(f"| {a['name']} | synthetic test | {pct(s['type_acc'])} | {pct(s['intent_acc'])} | {pct(s['amount_acc'])} | {pct(s['category_acc'])} | **{pct(s['end_to_end'])}** |")
        lines.append(f"| | hand-written | {pct(h['type_acc'])} | – | {pct(h['amount_acc'])} | {pct(h['category_acc'])} | **{pct(h['end_to_end'])}** |")
    lines += ["", "| Approach | Model size | Latency / message | Training time |", "|---|---|---|---|"]
    for a in r["approaches"]:
        lines.append(f"| {a['name']} | {a['size_mb']} MB | {a['latency_ms']} ms | {a['train_seconds']} s |")
    for a in r["approaches"]:
        if "ner" in a:
            n = a["ner"]
            lines += ["", f"spaCy NER on synthetic test (exact span match): precision {pct(n['precision'])}, recall {pct(n['recall'])}, F1 **{pct(n['f1'])}**. Entities skipped in training because of token misalignment: {a['misaligned_entities_skipped']}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
