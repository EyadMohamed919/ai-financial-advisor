"""Top level of the chatbot: route the message first, then hand it to the right component.

  message ──► Router ──► transaction    → Assistant (amount rules + TF-IDF + spaCy) → saved to the DB  "Classifier"
                     ├─► spending_query → computed from the user's records in the DB
                     ├─► forecast       → run-rate projection from the DB                            "Forecast"
                     ├─► market         → exchange-rate API, or Gemini + Google Search for prices     "APIs"
                     ├─► advice         → Gemini with the user's records from the DB (or rules)       "LLM"
                     ├─► manage         → undo / delete in the DB
                     └─► greeting, out_of_scope, unsure → fixed replies, or Gemini when configured
"""
import re

from . import db, handlers, llm
from .normalize import normalize
from .pipeline import Assistant
from .router import ROUTE_AR, Router


class Orchestrator:
    def __init__(self, use_db=True):
        """use_db=False keeps everything in memory (the caller passes `ledger`), e.g. for tests and notebooks."""
        self.router = Router().load()
        self.assistant = Assistant()
        self.use_db = use_db

    def handle(self, text, ledger=None, draft=None, category=None):
        if ledger is None:
            ledger = db.ledger() if self.use_db else []
        if draft:  # answering a follow-up question ("كان بكام؟"): stay in the transaction flow
            out = self.assistant.follow_up(draft, text=text, category=category)
            return self._save({**out, "route": "transaction", "route_ar": ROUTE_AR["transaction"], "route_confidence": None}, text)

        if re.search(handlers.IDENTITY_RE, normalize(text)):  # "انت جيميناي؟" whatever route the model picks
            return {"route": "greeting", "route_ar": ROUTE_AR["greeting"], "route_confidence": None,
                    "status": "not_applicable", "reply": handlers.IDENTITY_REPLY, "extracted_data": {"identity_question": True}}
        r = self.router.predict(text)
        route = r["route"]
        meta = {"route": route, "route_ar": ROUTE_AR[route], "route_confidence": round(r["confidence"], 3)}
        weak_greeting = route == "greeting" and r["confidence"] < 0.5  # real greetings score high ("ازيك" 0.83)
        if r["unsure"] or route == "out_of_scope" or (weak_greeting and llm.enabled()):
            if llm.enabled():
                result = handlers.general(text, ledger)
                return {**meta, "status": "answered", "reply": result["reply"], "extracted_data": {**result["data"], "router": r}}
            reply = handlers.UNSURE_REPLY if r["unsure"] else handlers.OUT_OF_SCOPE_REPLY
            return {**meta, "status": "not_applicable", "reply": reply, "extracted_data": {"router": r}}
        if route == "transaction":
            return self._save({**meta, **self.assistant.handle(text)}, text)
        if route == "greeting":
            result = handlers.greeting(text)
            return {**meta, "status": "not_applicable", "reply": result["reply"], "extracted_data": result["data"]}
        result = getattr(handlers, route)(text, ledger)
        action = result.get("action")
        if action and action["type"] == "delete" and self.use_db:
            db.delete(action["id"])
        return {**meta, "status": "answered", "reply": result["reply"], "extracted_data": result["data"], "action": action}

    def _save(self, out, text):
        """Persist a complete transaction (status "confirmed") and return the saved row with the reply."""
        if out.get("status") == "confirmed" and self.use_db:
            out["saved"] = db.add(out["extracted_data"], text)
        return out
