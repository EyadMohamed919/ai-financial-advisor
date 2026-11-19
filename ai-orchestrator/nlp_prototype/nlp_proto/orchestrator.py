"""Top level of the chatbot: route the message first, then hand it to the right component.

  message ──► Router ──► transaction    → Assistant (amount rules + TF-IDF + spaCy)   "Classifier"
                     ├─► spending_query → computed from the user's records
                     ├─► forecast       → run-rate projection                         "Forecast"
                     ├─► market         → exchange-rate API (+ LLM for concepts)      "APIs"
                     ├─► advice         → LLM with an aggregated summary (or rules)   "LLM"
                     ├─► manage         → undo / delete
                     └─► greeting, out_of_scope, unsure → fixed replies
"""
from . import handlers
from .pipeline import Assistant
from .router import ROUTE_AR, Router


class Orchestrator:
    def __init__(self):
        self.router = Router().load()
        self.assistant = Assistant()

    def handle(self, text, ledger=None, draft=None, category=None):
        ledger = ledger or []
        if draft:  # answering a follow-up question ("كان بكام؟"): stay in the transaction flow
            out = self.assistant.follow_up(draft, text=text, category=category)
            return {**out, "route": "transaction", "route_ar": ROUTE_AR["transaction"], "route_confidence": None}

        r = self.router.predict(text)
        route = r["route"]
        meta = {"route": route, "route_ar": ROUTE_AR[route], "route_confidence": round(r["confidence"], 3)}
        if r["unsure"]:
            return {**meta, "status": "not_applicable", "reply": handlers.UNSURE_REPLY, "extracted_data": {"router": r}}
        if route == "transaction":
            out = self.assistant.handle(text)
            if out["status"] == "not_applicable":  # router said transaction but the extractor disagrees
                return {**meta, **out}
            return {**meta, **out}
        if route == "greeting":
            return {**meta, "status": "not_applicable", "reply": handlers.GREETING_REPLY, "extracted_data": {}}
        if route == "out_of_scope":
            return {**meta, "status": "not_applicable", "reply": handlers.OUT_OF_SCOPE_REPLY, "extracted_data": {}}
        result = getattr(handlers, route)(text, ledger)
        return {**meta, "status": "answered", "reply": result["reply"], "extracted_data": result["data"],
                "action": result.get("action")}
