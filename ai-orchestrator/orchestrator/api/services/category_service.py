import re
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "ml_models"

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
UNSURE_BELOW = 0.35


def normalize(text: str) -> str:
    text = str(text).translate(AR_DIGITS)
    text = re.sub("[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)  # tashkeel + tatweel
    return text.lower().strip()


def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)


def run_prediction(model, description: str, unsure_below: float = UNSURE_BELOW) -> dict:
    clean = normalize(description)
    probs = model.predict_proba([clean])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])

    category, confidence = ranked[0]
    note = ""

    # if confidence < unsure_below:
    #     category = "unsure"
    #     note = f"model's best guess was {ranked[0][0]}"

    return {
        "category": str(category),
        "category_confidence": round(float(confidence), 4),
        "category_top_3": [
            {"category": str(name), 
             "confidence": round(float(prob), 4)}
            for name, prob in ranked[:3]
        ],
    }