"""Validate data/router/*.jsonl (router training data). Usage: python scripts/validate_router_data.py [--min N]"""
import argparse
import collections
import glob
import json
import os
import sys

ROUTES = {"transaction", "spending_query", "forecast", "market", "advice", "manage", "greeting", "out_of_scope"}
DIALECTS = {"ar-eg", "franco", "en", "mixed"}
ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=0)
    args = ap.parse_args()
    existing = {" ".join(json.loads(l)["text"].split()) for f in glob.glob(os.path.join(ROOT, "train", "*.jsonl"))
                for l in open(f, encoding="utf-8") if l.strip()}
    existing_ids = {json.loads(l)["id"] for f in glob.glob(os.path.join(ROOT, "train", "*.jsonl"))
                    for l in open(f, encoding="utf-8") if l.strip()}
    recs, bad, seen = [], 0, collections.Counter()
    for f in sorted(glob.glob(os.path.join(ROOT, "router", "*.jsonl"))):
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if not line.strip():
                continue
            where = f"{os.path.basename(f)}:{n}"
            try:
                r = json.loads(line)
            except json.JSONDecodeError as ex:
                print(f"{where}: invalid JSON ({ex})"); bad += 1; continue
            errs = []
            if "relabel_of" in r:  # route label for an existing training record
                if set(r) != {"relabel_of", "route"}:
                    errs.append(f"relabel keys {sorted(r)}")
                elif r["relabel_of"] not in existing_ids:
                    errs.append(f"unknown id {r['relabel_of']}")
            else:
                if set(r) != {"id", "text", "route", "dialect"}:
                    errs.append(f"keys {sorted(r)}")
                else:
                    t = " ".join(r["text"].split())
                    if not t:
                        errs.append("empty text")
                    if t in existing:
                        errs.append("text duplicates the transaction dataset")
                    seen[t] += 1
                    if r["dialect"] not in DIALECTS:
                        errs.append(f"bad dialect {r['dialect']}")
            if r.get("route") not in ROUTES:
                errs.append(f"bad route {r.get('route')}")
            for e in errs:
                print(f"{where}: {e}")
            bad += bool(errs)
            recs.append(r)
    dups = sum(c - 1 for c in seen.values() if c > 1)
    new = [r for r in recs if "text" in r]
    print(f"\nnew records: {len(new)}  relabels: {len(recs) - len(new)}  invalid: {bad}  duplicate texts: {dups}")
    print("routes (new):", dict(collections.Counter(r.get("route") for r in new).most_common()))
    print("routes (relabels):", dict(collections.Counter(r.get("route") for r in recs if "relabel_of" in r).most_common()))
    print("dialects:", dict(collections.Counter(r.get("dialect") for r in new).most_common()))
    if bad or dups or len(new) < args.min:
        sys.exit(1)


if __name__ == "__main__":
    main()
