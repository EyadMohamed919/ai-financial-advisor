"""Train the demo models on all 2,000 synthetic records. Usage: python -m nlp_proto.train"""
from .data import load_synthetic
from .spacy_model import SpacyApproach
from .tfidf import TfidfApproach


def main():
    recs = load_synthetic()
    TfidfApproach().train(recs)
    print("trained TF-IDF classifier")
    SpacyApproach().train(recs)
    print("trained spaCy NER (takes about 3 minutes on a laptop CPU)")


if __name__ == "__main__":
    main()
