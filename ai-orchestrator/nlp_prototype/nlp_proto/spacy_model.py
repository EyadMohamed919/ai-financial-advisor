"""Approach C: spaCy pipelines (NER + intent textcat, and a category textcat), CPU-only.

The NER finds which span is the amount, the item, the merchant, etc. The span is then turned
into a number by amounts.parse_number_text, with the regex extractor as a fallback.
"""
import os
import random
import shutil

import spacy
from spacy.training import Example
from spacy.util import compile_infix_regex, fix_random_seed, minibatch

from .amounts import USD_RE, extract_amount, parse_number_text
from .data import MODELS, TX_INTENTS, entity_offsets
from .normalize import to_western_digits

AR = r"ء-يـ"
DIGIT = r"0-9٠-٩"


def make_nlp():
    nlp = spacy.blank("ar")
    # Split "ب1200" into "ب" + "1200" so the amount can be its own entity.
    infixes = list(nlp.Defaults.infixes) + [rf"(?<=[{AR}])(?=[{DIGIT}])", rf"(?<=[{DIGIT}])(?=[{AR}])"]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    return nlp


def _train(nlp, examples, epochs, seed=0):
    fix_random_seed(seed)
    optimizer = nlp.initialize(lambda: examples)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(examples)
        losses = {}
        for batch in minibatch(examples, size=16):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
    return nlp


class SpacyApproach:
    name = "C. spaCy (NER + textcat)"
    main_dir = os.path.join(MODELS, "spacy_main")
    cat_dir = os.path.join(MODELS, "spacy_category")

    def __init__(self, epochs=15):
        self.epochs = epochs
        self.misaligned = 0

    def train(self, recs):
        main = make_nlp()
        main.add_pipe("ner")
        main.add_pipe("textcat")  # exclusive: exactly one intent per message
        examples = []
        for r in recs:
            doc = main.make_doc(r["text"])
            ents = []
            for start, end, label in entity_offsets(r):
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is None:
                    self.misaligned += 1  # span boundary falls inside a token; skip that entity
                else:
                    ents.append((start, end, label))
            cats = {i: float(i == r["intent"]) for i in sorted({x["intent"] for x in recs})}
            examples.append(Example.from_dict(doc, {"entities": ents, "cats": cats}))
        self.main = _train(main, examples, self.epochs)

        cat_nlp = make_nlp()
        cat_nlp.add_pipe("textcat")
        tx = [r for r in recs if r["intent"] in TX_INTENTS and r["category"]]
        labels = sorted({f"{r['intent']}|{r['category']}" for r in tx})
        cat_examples = [Example.from_dict(cat_nlp.make_doc(r["text"]),
                                          {"cats": {l: float(l == f"{r['intent']}|{r['category']}") for l in labels}})
                        for r in tx]
        self.cat = _train(cat_nlp, cat_examples, self.epochs)

        for d, nlp in ((self.main_dir, self.main), (self.cat_dir, self.cat)):
            shutil.rmtree(d, ignore_errors=True)
            nlp.to_disk(d)

    def load(self):
        self.main, self.cat = spacy.load(self.main_dir), spacy.load(self.cat_dir)
        return self

    def size_bytes(self):
        return sum(os.path.getsize(os.path.join(dp, f)) for d in (self.main_dir, self.cat_dir)
                   for dp, _, files in os.walk(d) for f in files)

    def predict(self, text):
        doc = self.main(text)
        intent = max(doc.cats, key=doc.cats.get)
        entities = [{"label": e.label_, "text": e.text} for e in doc.ents]
        result = {"intent": intent, "category": None, "category_conf": None,
                  "amount": None, "currency": None, "span": None, "entities": entities}
        amount_ents = [e for e in doc.ents if e.label_ == "AMOUNT"]
        value = parse_number_text(amount_ents[0].text) if amount_ents else None
        if value:
            result.update(amount=value, currency="USD" if USD_RE.search(to_western_digits(text).lower()) else "EGP",
                          span=(amount_ents[0].start_char, amount_ents[0].end_char))
        else:
            result.update(extract_amount(text))  # fallback when NER finds no usable span
        if intent in TX_INTENTS:
            cats = self.cat(text).cats
            allowed = {k: v for k, v in cats.items() if k.startswith(intent + "|")}
            if allowed:
                best = max(allowed, key=allowed.get)
                result["category"] = best.split("|", 1)[1]
                result["category_conf"] = allowed[best] / sum(allowed.values())
        return result
