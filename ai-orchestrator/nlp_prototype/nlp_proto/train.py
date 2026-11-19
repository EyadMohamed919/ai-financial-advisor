"""Train the demo models (classifier, spaCy NER, router) on all data. Usage: python -m nlp_proto.train"""
from .data import load_synthetic
from .router import Router, load_router_data
from .spacy_model import SpacyApproach
from .tfidf import TfidfApproach


def main():
    recs = load_synthetic()
    TfidfApproach().train(recs)
    print("trained TF-IDF classifier")
    SpacyApproach().train(recs)
    print("trained spaCy NER (takes about 3 minutes on a laptop CPU)")
    Router().train(load_router_data())
    print("trained router")


if __name__ == "__main__":
    main()
