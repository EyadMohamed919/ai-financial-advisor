import re
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml_models" / "transaction_ner_model.joblib"

nlp = joblib.load(MODEL_PATH)

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text: str) -> str:
    text = str(text).translate(AR_DIGITS)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_expense(text: str) -> dict:
    clean = normalize(text)
    doc = nlp(clean)

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]

    fields = {"ITEM": None, "MERCHANT": None, "DATE": None}
    for ent in doc.ents:
        if ent.label_ in fields and fields[ent.label_] is None:
            fields[ent.label_] = ent.text

    return {
        "item": fields["ITEM"] or "",
        "merchant": fields["MERCHANT"],
        "date_ref": fields["DATE"],
        "entities": entities,
    }