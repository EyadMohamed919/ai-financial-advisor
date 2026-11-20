"""Handlers for the non-transaction routes. Each returns {"reply", "data", optional "action"}.

`ledger` is the user's recorded transactions: [{"type", "amount", "currency", "category", "label", "date"}].
In the prototype the browser sends it with every message; in the real app it comes from the
database, scoped to the authenticated user only (FR-026, FR-029).
"""
import calendar
import datetime as dt
import json
import re
import time
import urllib.request

from . import llm
from .normalize import normalize
from .pipeline import CATEGORY_AR, _money
from .rules import ALLOWED, CATEGORY_KEYWORDS

DISCLAIMER = "ℹ️ دي معلومات إرشادية مش استشارة مالية مرخصة."
MIN_TRANSACTIONS = 5  # FR-022: fewer than five records -> insufficient-data notice


# ---------- helpers ----------
def _period(t, today):
    """Return (start, end, label) for the time phrase in the message. Default: this month."""
    month_start = today.replace(day=1)
    if re.search(r"النهارده|النهاردة|today|ennharda|elnaharda", t):
        return today, today, "النهارده"
    if re.search(r"امبارح|yesterday|embare7", t):
        d = today - dt.timedelta(days=1)
        return d, d, "امبارح"
    if re.search(r"الاسبوع اللي فات|الاسبوع الماضي|last week", t):
        return today - dt.timedelta(days=13), today - dt.timedelta(days=7), "الأسبوع اللي فات"
    if re.search(r"الاسبوع|اسبوع|this week|week", t):
        return today - dt.timedelta(days=6), today, "آخر 7 أيام"
    if re.search(r"الشهر اللي فات|الشهر الماضي|last month|el shahr elly fat", t):
        end = month_start - dt.timedelta(days=1)
        return end.replace(day=1), end, "الشهر اللي فات"
    if re.search(r"السنه|السنة|this year|year", t):
        return today.replace(month=1, day=1), today, "من أول السنة"
    return month_start, today, "الشهر ده"


def _in_period(entry, start, end):
    try:
        d = dt.date.fromisoformat(entry.get("date", "")[:10])
    except ValueError:
        return False
    return start <= d <= end


def _egp(entry):
    return entry["amount"] if entry.get("currency", "EGP") == "EGP" else 0  # prototype totals count EGP only


def _category_in(t):
    for cat, pattern in CATEGORY_KEYWORDS:
        if cat in ALLOWED["log_expense"] and re.search(pattern, t):
            return cat
    return None


def _month_summary(ledger, today):
    start = today.replace(day=1)
    month = [e for e in ledger if _in_period(e, start, today)]
    by_cat = {}
    for e in month:
        if e["type"] == "expense":
            by_cat[e["category"]] = by_cat.get(e["category"], 0) + _egp(e)
    income = sum(_egp(e) for e in month if e["type"] == "income")
    return month, income, dict(sorted(by_cat.items(), key=lambda kv: -kv[1]))


# ---------- spending_query (FR-018) ----------
def spending_query(text, ledger, today=None):
    today = today or dt.date.today()
    t = normalize(text)
    if not ledger:
        return {"reply": "لسه مفيش معاملات مسجلة عشان أحسبلك. سجّل شوية مصاريف الأول 🙂", "data": {"records": 0}}
    start, end, label = _period(t, today)
    entries = [e for e in ledger if _in_period(e, start, end)]
    if re.search(r"دخل|قبضت|income|earned|مرتب", t):
        total = sum(_egp(e) for e in entries if e["type"] == "income")
        return {"reply": f"دخلك {label}: {_money(total, 'EGP')}.", "data": {"period": label, "income": total}}
    expenses = [e for e in entries if e["type"] == "expense"]
    if re.search(r"اكبر|اكتر|اعلي|most|biggest|top", t):
        by_cat = {}
        for e in expenses:
            by_cat[e["category"]] = by_cat.get(e["category"], 0) + _egp(e)
        if not by_cat:
            return {"reply": f"مفيش مصاريف مسجلة {label}.", "data": {"period": label}}
        cat, amount = max(by_cat.items(), key=lambda kv: kv[1])
        share = amount / sum(by_cat.values())
        return {"reply": f"أكبر بند {label}: {CATEGORY_AR.get(cat, cat)} بـ {_money(amount, 'EGP')} ({share:.0%} من مصاريفك).",
                "data": {"period": label, "top_category": cat, "amount": amount}}
    cat = _category_in(t)
    if cat:
        expenses = [e for e in expenses if e["category"] == cat]
    total = sum(_egp(e) for e in expenses)
    what = f" على {CATEGORY_AR[cat]}" if cat else ""
    if not expenses:
        return {"reply": f"مفيش مصاريف{what} مسجلة {label}.", "data": {"period": label, "category": cat, "total": 0}}
    return {"reply": f"صرفت{what} {label}: {_money(total, 'EGP')} ({len(expenses)} معاملة).",
            "data": {"period": label, "category": cat, "total": total, "count": len(expenses)}}


# ---------- forecast ----------
def forecast(text, ledger, today=None):
    today = today or dt.date.today()
    if len(ledger) < MIN_TRANSACTIONS:
        return {"reply": f"محتاج {MIN_TRANSACTIONS} معاملات على الأقل عشان أتوقع. عندك {len(ledger)} لحد دلوقتي.",
                "data": {"records": len(ledger), "insufficient_data": True}}
    month, income, by_cat = _month_summary(ledger, today)
    spent = sum(by_cat.values())
    days = calendar.monthrange(today.year, today.month)[1]
    projected = spent / today.day * days
    reply = (f"لو فضلت بنفس المعدل، هتصرف حوالي {_money(round(projected, -1), 'EGP')} الشهر ده "
             f"(صرفت {_money(spent, 'EGP')} في {today.day} يوم).")
    if income:
        reply += f" ودخلك المسجل {_money(income, 'EGP')}، يعني هيفضل معاك حوالي {_money(round(income - projected, -1), 'EGP')}."
    if today.day < 7:
        reply += " (التوقع لسه مش دقيق لأننا في أول الشهر.)"
    return {"reply": reply + "\n" + DISCLAIMER,
            "data": {"spent_so_far": spent, "projected_month": round(projected, 2), "income": income, "method": "linear run-rate"}}


# ---------- market (FR-023, NFR-P03) ----------
CURRENCIES = [("USD", r"دولار|usd|\$|dollar|dolar"), ("EUR", r"يورو|euro|eur"), ("SAR", r"ريال|sar|riyal"),
              ("GBP", r"استرليني|باوند|gbp|sterling"), ("AED", r"درهم|aed|dirham"), ("KWD", r"دينار|kwd|dinar")]
OTHER_ASSETS = [("stocks", r"سهم|اسهم|البورصه|egx|stock|share|اسهمي"), ("gold", r"دهب|ذهب|gold|عيار"),
                ("inflation", r"تضخم|inflation"), ("interest", r"فايده|فائده|فوايد|شهاده|شهادات|interest|cbe|المركزي")]
FX_WORDS = r"سعر الصرف|العمله|العملات|عمله|exchange|fx|forex"
ASSET_AR = {"stocks": "أسعار الأسهم والبورصة", "gold": "أسعار الدهب", "inflation": "أرقام التضخم", "interest": "أسعار الفايدة والشهادات"}
_rates_cache = {"at": 0, "data": None}


def _usd_rates():
    """Live exchange rates from open.er-api.com (free, no key), cached for an hour. None on failure."""
    if _rates_cache["data"] and time.time() - _rates_cache["at"] < 3600:
        return _rates_cache["data"]
    try:
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("result") != "success" or "EGP" not in data.get("rates", {}):
            return None
        _rates_cache.update(at=time.time(), data=data)
        return data
    except Exception:
        return None


def market(text, ledger=None, today=None):
    t = normalize(text)
    parts, data = [], {"source": None}
    currencies = [code for code, pat in CURRENCIES if re.search(pat, t)]
    assets = [a for a, pat in OTHER_ASSETS if re.search(pat, t)]
    fx_question = bool(currencies) or bool(re.search(FX_WORDS, t))
    if not currencies and not assets and not fx_question:
        # A price question about something we have no data for (a car, a phone...): say so instead of
        # answering with an unrelated exchange rate.
        parts.append("مقدرش أجيب سعر الحاجة دي دلوقتي 🙏 البيانات المتاحة في النسخة التجريبية هي أسعار العملات بس "
                     "(جرّب: «الدولار بكام؟»).")
        data["unknown_asset"] = True
    if fx_question:
        rates = _usd_rates()
        if rates:
            egp = rates["rates"]["EGP"]
            for code in currencies or ["USD"]:
                per_unit = egp / rates["rates"][code]
                parts.append(f"1 {code} ≈ {per_unit:,.2f} جنيه")
            updated = rates.get("time_last_update_utc", "")[:16]
            data.update(source="open.er-api.com", updated=updated, currencies=currencies or ["USD"])
            parts.append(f"(المصدر: open.er-api.com، آخر تحديث: {updated})")
        else:
            parts.append("مش قادر أوصل لأسعار الصرف دلوقتي. هكمّل من غير بيانات السوق.")  # NFR-P03 fallback
            data["economic_data_unavailable"] = True
    for a in assets:
        parts.append(f"{ASSET_AR[a]} لسه مش متوصلة في النسخة التجريبية.")
    data["assets_requested"] = assets
    if (assets or data.get("unknown_asset")) and llm.enabled():
        answer = llm.generate(
            f"سؤال المستخدم: {text}",
            "انت مساعد مالي مصري. اشرح بالعامية المصرية في 3 جمل بالكتير المفاهيم العامة اللي تساعد المستخدم يفهم السؤال. "
            "ممنوع تذكر أي أسعار أو أرقام حالية لأنك مش متأكد منها، وممنوع توصي بشراء أو بيع أصل معين.")
        if answer:
            parts.append(answer)
            data["llm"] = True
    return {"reply": "\n".join(parts) + "\n" + DISCLAIMER, "data": data}


# ---------- advice (FR-022, FR-025, FR-029) ----------
def advice(text, ledger, today=None):
    today = today or dt.date.today()
    month, income, by_cat = _month_summary(ledger or [], today)
    # Only aggregated numbers go to the LLM: no raw messages, merchants or personal details (FR-029).
    summary = {"income_this_month": income, "expenses_by_category_this_month": {CATEGORY_AR.get(k, k): v for k, v in by_cat.items()},
               "transactions_recorded": len(ledger or [])}
    data = {"summary_sent_to_llm": summary, "llm": False}
    answer = None
    if llm.enabled():
        answer = llm.generate(
            f"سؤال المستخدم: {text}\nملخص بياناته (بالجنيه): {json.dumps(summary, ensure_ascii=False)}",
            "انت مساعد مالي مصري لمستخدمين معلوماتهم المالية بسيطة. رد بالعامية المصرية في 4 جمل بالكتير، "
            "بنصايح عملية ومحددة مبنية على الملخص لو مفيد. ماتوصيش بمنتجات استثمار معينة ولا تقول أرقام سوق.")
        data["llm"] = bool(answer)
    if not answer:
        if len(ledger or []) >= MIN_TRANSACTIONS and by_cat:
            cat, amount = next(iter(by_cat.items()))
            share = amount / sum(by_cat.values())
            answer = (f"أكبر بند عندك الشهر ده {CATEGORY_AR.get(cat, cat)} ({share:.0%} من مصاريفك). "
                      f"لو قللته 10% هتوفر حوالي {_money(round(amount * 0.1, -1), 'EGP')} في الشهر. "
                      "وقاعدة بسيطة تبدأ بيها: حوّش 20% من دخلك أول ما تقبض قبل ما تصرف.")
        else:
            answer = "قاعدة بسيطة تبدأ بيها: قسّم دخلك 50% للأساسيات، و30% للمصاريف الشخصية، و20% للتحويش. ولما تسجل مصاريف أكتر، هقدر أديك نصايح على حسب صرفك."
        if not llm.enabled():
            data["note"] = "LLM not configured (set GEMINI_API_KEY); this is the built-in rule-based advice."
    return {"reply": answer + "\n" + DISCLAIMER, "data": data}


# ---------- manage (FR-016) ----------
def manage(text, ledger, today=None):
    t = normalize(text)
    if re.search(r"غير|عدل|edit|change|correct|ghayar|3adel", t):
        return {"reply": "التعديل من الشات جاي في النسخة الجاية. دلوقتي امسح المعاملة من اللوحة وسجلها تاني.", "data": {"action": "edit_not_supported"}}
    if not ledger:
        return {"reply": "مفيش معاملات مسجلة أمسحها.", "data": {}}
    last = ledger[0]
    return {"reply": f"مسحت آخر معاملة: {last.get('label') or CATEGORY_AR.get(last['category'], '')} – {_money(last['amount'], last.get('currency', 'EGP'))} ↩️",
            "data": {"deleted": last}, "action": {"type": "delete", "id": last.get("id")}}


IDENTITY_RE = r"انت مين|انت ايه|مين انت|انت بوت|بوت ولا|جيميناي|جيمناي|gemini|chatgpt|شات جي بي تي|who are you|what are you|are you (?:a )?(?:bot|ai|gemini|human)|enta meen|inta meen"
IDENTITY_REPLY = ("أنا المساعد المالي بتاع المشروع 🤖، نسخة تجريبية. بفهم رسايلك بموديلات صغيرة متدربة على مصاريف بالمصري، "
                  "مش Gemini ولا ChatGPT. الـ LLM (Gemini) بيتستخدم بس في النصايح وشرح أسئلة الأسواق لو متوصل.")


def greeting(text, ledger=None, today=None):
    if re.search(IDENTITY_RE, normalize(text)):
        return {"reply": IDENTITY_REPLY, "data": {"identity_question": True}}
    return {"reply": GREETING_REPLY, "data": {}}


GREETING_REPLY = "أهلاً بيك 👋 أقدر أسجلك مصاريفك ودخلك، وأقولك صرفت كام، وأتوقع مصاريف الشهر، وأديك أسعار العملات ونصايح بسيطة."
OUT_OF_SCOPE_REPLY = "ده برة اللي أقدر أساعد فيه 🙏 أنا متخصص في فلوسك: تسجيل المصاريف، والحسابات، والنصايح، وأسعار العملات."
UNSURE_REPLY = "مش متأكد قصدك 🤔 عاوز تسجل مصروف، ولا تسأل عن مصاريفك، ولا عاوز نصيحة؟"
