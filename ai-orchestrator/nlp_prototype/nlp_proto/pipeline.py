"""The recommended pipeline: what the chatbot would actually run.

- amount and currency: rules (amounts.py), because numbers are deterministic
- intent and category: TF-IDF + logistic regression (best classifier in the comparison)
- item, merchant, date, person: spaCy NER
- reply: fixed Egyptian-Arabic templates, so the bot never invents text

The output matches the ERD: `extracted_data` is stored in chat_messages.extracted_data and
`status` maps to chat_messages.extraction_status. A complete message is saved directly (`confirmed`);
the bot only asks when the amount is missing or the category is uncertain.
"""
from .amounts import extract_amount
from .data import TX_INTENTS
from .rules import ALLOWED
from .spacy_model import SpacyApproach
from .tfidf import TfidfApproach

CATEGORY_CONFIDENCE_MIN = 0.5  # below this the bot asks instead of guessing

TYPE_AR = {"log_expense": "مصروف", "log_income": "دخل", "log_savings": "ادخار", "log_transfer": "تحويل"}
CATEGORY_AR = {
    "Food": "أكل", "Transport": "مواصلات", "Housing": "سكن", "Utilities": "فواتير", "Subscriptions": "اشتراكات",
    "Entertainment": "خروج وترفيه", "Shopping": "مشتريات", "Health": "صحة", "Education": "تعليم", "Debt": "أقساط وديون",
    "Other": "أخرى", "Salary": "مرتب", "Freelance": "فريلانس", "Gift": "هدية", "Refund": "مرتجع",
    "Savings": "ادخار", "Transfer": "تحويل", "Lending": "سلفة",
}


def _money(amount, currency):
    value = f"{amount:,.0f}" if amount == int(amount) else f"{amount:,.2f}"
    return f"{value} {'دولار' if currency == 'USD' else 'جنيه'}"


class Assistant:
    def __init__(self):
        self.classifier = TfidfApproach().load()
        self.ner = SpacyApproach().load()

    def handle(self, text):
        cls = self.classifier.predict(text)
        intent = cls["intent"]
        if intent not in TX_INTENTS:
            replies = {
                "greeting": "أهلاً بيك 👋 ابعتلي أي مصروف أو دخل وأنا أسجلهولك. مثلاً: «دفعت 120 أكل»",
                "question": "الأسئلة عن مصاريفك جاية في النسخة الجاية 🙏 دلوقتي أقدر أسجلك أي مصروف أو دخل.",
            }
            return {"status": "not_applicable", "extracted_data": {"intent": intent},
                    "reply": replies.get(intent, "مش فاهم قصدك 🤔 اكتب المبلغ وصرفته على إيه، زي «جبت عيش ب 20».")}

        ents = {}
        for e in self.ner.main(text).ents:
            ents.setdefault(e.label_.lower(), e.text)
        amount = extract_amount(text)
        data = {"intent": intent, "type": intent.removeprefix("log_"), "amount": amount["amount"],
                "currency": amount["currency"], "category": cls["category"],
                "category_confidence": round(cls["category_conf"], 3) if cls["category_conf"] is not None else None,
                "item": ents.get("item"), "merchant": ents.get("merchant"), "date": ents.get("date"),
                "person": ents.get("person")}

        return self._finish(data)

    def follow_up(self, draft, text=None, category=None):
        """Complete a `needs_info` draft with the user's next answer (an amount or a chosen category)."""
        data = dict(draft)
        if category is not None and category in ALLOWED.get(data["intent"], ()):
            data["category"], data["category_confidence"] = category, 1.0
        elif text is not None and "amount" in data.get("missing", []):
            amount = extract_amount(text)
            if amount["amount"] is None:
                return {"status": "needs_info", "extracted_data": data, "reply": "مش لاقي مبلغ 🤔 اكتب الرقم بس، زي «350»."}
            data["amount"], data["currency"] = amount["amount"], amount["currency"]
        else:
            return self.handle(text or "")
        return self._finish(data)

    def _finish(self, data):
        intent = data["intent"]
        what = data["item"] or data["merchant"] or "ده"
        if data["amount"] is None:
            data["missing"] = ["amount"]
            return {"status": "needs_info", "extracted_data": data, "reply": f"تمام، {what} كان بكام؟"}
        if data["category"] is None or (data["category_confidence"] or 0) < CATEGORY_CONFIDENCE_MIN:
            data["missing"] = ["category"]
            guess = f" (أقرب تخمين: {CATEGORY_AR[data['category']]})" if data["category"] else ""
            options = [{"value": c, "label": CATEGORY_AR[c]} for c in sorted(ALLOWED[intent])]
            return {"status": "needs_info", "extracted_data": data, "category_options": options,
                    "reply": f"{_money(data['amount'], data['currency'])} على {what}… أحطها تحت أنهي فئة؟{guess}"}

        data["missing"] = []
        details = " ".join(x for x in (data["item"], f"من {data['merchant']}" if data["merchant"] else None, data["date"]) if x)
        reply = (f"اتسجل ✅ {TYPE_AR[intent]}: {_money(data['amount'], data['currency'])} – {CATEGORY_AR[data['category']]}"
                 f"{f' ({details})' if details else ''}")
        return {"status": "confirmed", "extracted_data": data, "reply": reply}
