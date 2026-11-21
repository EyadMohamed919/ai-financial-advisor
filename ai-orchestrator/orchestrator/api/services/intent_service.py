import re
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml_models" / "intent_model.joblib"

model = joblib.load(MODEL_PATH)

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

FUTURE_RE = re.compile(
    r"(?:^|\s)(?:ه|هي|هن)(?:صرف|حوش|وفر|دفع|فضل|تبقي|كفي|خلص|قبض|يفضل|يتبقي|يكفي|يخلص)\w*"
    r"|\bwill i\b|\bam i going to\b|\bhow much will\b|\bha(?:sraf|7awesh|dfa3)\b|\bhayfdal\b",
    re.I,
)
UNSURE_BELOW = 0.35


def normalize(text: str) -> str:
    text = str(text).translate(AR_DIGITS)
    text = re.sub("[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)  # tashkeel + tatweel
    return text.lower().strip()


def predict_intent(message: str) -> dict:
    clean = normalize(message)
    probs = model.predict_proba([clean])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])

    intent, confidence = ranked[0]
    note = ""

    if intent == "spending_query" and FUTURE_RE.search(clean):
        intent = "forecast"
        note = "changed from spending_query by the future-tense rule"

    if confidence < UNSURE_BELOW:
        intent = "unsure"
        note = f"model's best guess was {ranked[0][0]}"

    return {
        "message": message,
        "intent": str(intent),
        "confidence": round(float(confidence), 4),
        "note": note,
        "top_3": [
            {"intent": str(name), "confidence": round(float(prob), 4)}
            for name, prob in ranked[:3]
        ],
    }