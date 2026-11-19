"""Rule-based amount extraction: digits, Arabic digits, "9.5k" and Egyptian number words.

Numbers are deterministic, so they are parsed with rules instead of a learned model.
The learned models decide *which* span is the amount (spaCy NER); this module turns
any span into a number.
"""
import re

from .normalize import normalize_letters, to_western_digits

UNITS = {
    "واحد": 1, "واحده": 1, "اتنين": 2, "اثنين": 2, "تلاته": 3, "تلات": 3, "تلت": 3, "ثلاثه": 3,
    "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7,
    "تمانيه": 8, "تمان": 8, "تمن": 8, "تسعه": 9, "تسع": 9, "عشره": 10, "عشر": 10,
    "حداشر": 11, "اتناشر": 12, "تلتاشر": 13, "اربعتاشر": 14, "اربعطاشر": 14, "خمستاشر": 15,
    "خمسطاشر": 15, "ستاشر": 16, "سبعتاشر": 17, "سبعطاشر": 17, "تمنتاشر": 18, "تمانتاشر": 18,
    "تسعتاشر": 19, "تسعطاشر": 19,
    "عشرين": 20, "تلاتين": 30, "اربعين": 40, "خمسين": 50, "ستين": 60, "سبعين": 70,
    "تمانين": 80, "تسعين": 90,
}
HUNDREDS = {
    "ميه": 100, "ميت": 100, "مايه": 100, "ميتين": 200, "تلتميه": 300, "ربعميه": 400,
    "خمسميه": 500, "ستميه": 600, "سبعميه": 700, "تمنميه": 800, "تسعميه": 900,
    "متين": 200, "اربعميه": 400, "ثلاثميه": 300, "ثمانميه": 800, "تمانميه": 800,
}
THOUSANDS = {"الف", "الاف", "تلاف", "الالاف", "باكو"}  # باكو: Egyptian slang for 1,000
MILLIONS = {"مليون", "ملايين", "ارنب"}  # ارنب: slang for 1,000,000
HALF, QUARTER = {"نص", "نصف"}, {"ربع"}
VOCAB = set(UNITS) | set(HUNDREDS) | THOUSANDS | MILLIONS | HALF | QUARTER | {"الفين"}

CURRENCY_RE = r"(?:جنيه|جنيها|جنية|جنيهات|ج\.م|ج|egp|le|l\.e|pound|pounds|دولار|usd|\$)"
USD_RE = re.compile(r"\$|usd|دولار|dollar", re.I)
NUM_RE = re.compile(r"(?<![\d*/:.,a-z])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?k\b|\s?(?:الف|ألف|باكو))?(?![\d/:a-jl-z]|\.\d|%)", re.I)
HALF_THOUSAND_RE = re.compile(r"(?<![\d.])(\d+)\s?(?:و\s?)?نص\s?(?:الف|ألف|باكو)")
BEFORE_CUE = re.compile(r"(?:\bب\s?ـ?\s?|بمبلغ\s?|مبلغ\s?|\bfor\s|\bpaid\s|\bspent\s|" + CURRENCY_RE + r"\s?)$", re.I)
AFTER_CUE = re.compile(r"^\s?" + CURRENCY_RE + r"(?![a-z؀-ۿ])", re.I)
QUANTITY_AFTER = re.compile(r"^\s?(?:كيلو|علب|علبه|شهور|شهر|ايام|يوم|قطع|قطعه|حته|حتت|تيشيرت|تيشيرتات|مرات|مره|نفر|ساعات|ساعه|x\b|pcs|kg|months?|days?|items?)", re.I)
BALANCE_BEFORE = re.compile(r"(?:\bbal\b\.?|balance|رصيد|متاح|المتبقي|\bref\b\.?|رقم|otp|كود)[^\d]{0,22}$", re.I)


def _word_value(tokens):
    """Value of a run of Egyptian number words, e.g. ['ألف', 'وخمسمية'] -> 1500."""
    total, current, last_mult, seen = 0.0, 0.0, 1, False
    for raw in tokens:
        w = _strip_prefix(raw)
        if w in HALF or w in QUARTER:
            frac = 0.5 if w in HALF else 0.25
            if last_mult >= 1000:
                total += last_mult * frac
            elif last_mult == 100:
                current += 100 * frac
            else:
                current += frac
        elif w in UNITS:
            current += UNITS[w]
        elif w in HUNDREDS:
            current = current * 100 if w in ("ميه", "ميت", "مايه") and 0 < current < 10 else current + HUNDREDS[w]
            last_mult = 100
        elif w == "الفين":
            total += 2000
            last_mult = 1000
        elif w in THOUSANDS:
            total += (current or 1) * 1000
            current, last_mult = 0, 1000
        elif w in MILLIONS:
            total += (current or 1) * 1_000_000
            current, last_mult = 0, 1_000_000
        else:
            continue
        seen = True
    return total + current if seen else None


def _strip_prefix(token: str) -> str:
    w = normalize_letters(token).strip("،,.!?؟").replace("مائه", "ميه")
    for prefix in ("وب", "و", "ب"):
        if w.startswith(prefix) and w[len(prefix):] in VOCAB:
            return w[len(prefix):]
    return w


def parse_number_text(span: str):
    """Convert an amount span ('1,200', '١٢٠٠', '9.5k', 'ألف وخمسمية') to a float, or None."""
    t = to_western_digits(span).strip().lower()
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", t)
    if m:
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        rest = t[m.end():].strip()
        if re.match(r"(?:و\s?)?نص\s?(?:الف|ألف|باكو)", rest):
            return (value + 0.5) * 1000
        if rest.startswith(("k", "الف", "ألف", "باكو")):
            value *= 1000
        return value
    return _word_value(t.split())


def _word_runs(text: str):
    """Yield (start, end, value, n_tokens, has_multiplier) for runs of number words."""
    tokens = list(re.finditer(r"\S+", text))
    i = 0
    while i < len(tokens):
        if _strip_prefix(tokens[i].group()) in VOCAB:
            j = i
            while j + 1 < len(tokens) and (_strip_prefix(tokens[j + 1].group()) in VOCAB or tokens[j + 1].group() == "و"):
                j += 1
            run = [tok.group() for tok in tokens[i:j + 1]]
            value = _word_value(run)
            if value:
                words = {_strip_prefix(t) for t in run}
                has_mult = bool(words & (set(HUNDREDS) | THOUSANDS | MILLIONS | {"الفين"}))
                yield tokens[i].start(), tokens[j].end(), value, len(run), has_mult
            i = j + 1
        else:
            i += 1


def extract_amount(text: str):
    """Return {'amount', 'currency', 'span'} using rules only. amount is None when nothing is found."""
    t = to_western_digits(text).lower()  # same length as text, so offsets stay valid
    candidates = []
    for m in HALF_THOUSAND_RE.finditer(t):  # "2 ونص ألف" = 2500
        candidates.append((4, (float(m.group(1)) + 0.5) * 1000, m.start(), m.end()))
    for m in NUM_RE.finditer(t):
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        if m.group(3):
            value *= 1000
        before, after = t[max(0, m.start() - 30):m.start()], t[m.end():m.end() + 14]
        score = 0
        if BEFORE_CUE.search(before[-12:]):
            score += 3
        if AFTER_CUE.search(after):
            score += 3
        if m.group(3):
            score += 1
        if QUANTITY_AFTER.search(after):
            score -= 3
        if BALANCE_BEFORE.search(before):
            score -= 5
        if value == 0:
            continue
        candidates.append((score, value, m.start(), m.end()))
    if not candidates:
        for start, end, value, n_tokens, has_mult in _word_runs(text):
            near_currency = bool(AFTER_CUE.search(t[end:end + 14]))
            if n_tokens == 1 and not has_mult and not near_currency and value < 11:
                continue  # "واحد صاحبي" is not an amount
            candidates.append((2 if near_currency else 0, value, start, end))
    if not candidates:
        return {"amount": None, "currency": None, "span": None}
    score, value, start, end = max(candidates, key=lambda c: (c[0], c[1]))
    currency = "USD" if USD_RE.search(t) else "EGP"
    return {"amount": value, "currency": currency, "span": (start, end)}
