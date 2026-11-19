"""Approach A: keyword rules + rule-based amounts. No training. This is the baseline."""
import re

from .amounts import extract_amount
from .normalize import normalize

INTENT_KEYWORDS = [
    ("log_income", r"قبضت|مرتب|راتب|بونص|نزل(?:ي|لي)? في حسابي|استلمت|اتحولي|بعتلي|رجعلي|رجعتلي|استرجعت|refund|salary|bonus|received|got paid|ead+ya|عيديه|2abadt|2bdt|mortab"),
    ("log_savings", r"حوشت|حطيت .*(?:توفير|شهاده|حساب)|شهاده|جمعيه|ادخار|توفير|savings|saved|7awesht|gam3eya"),
    ("log_transfer", r"سلفت|حولت|تحويل|سحبت .*(?:atm|ماكينه)|سحب|lent|transfer|sallaft|7awelt"),
    ("log_expense", r"دفعت|اشتريت|جبت|صرفت|اشتركت|جددت|شحنت|طلبت|ركبت|سددت|خلصت|paid|bought|spent|dafa3t|eshtaret|gebt|saraft|اتغديت|فطرت|اتعشيت"),
]
QUESTION = re.compile(r"[?؟]|كام|ازاي|ليه|امتي|فين|ايه|هل |how|what|why|when")
GREETING = re.compile(r"^(?:اهلا|السلام|صباح|مساء|هاي|hi|hello|hey|ازيك|عامل ايه|شكرا|thanks)")

CATEGORY_KEYWORDS = [
    ("Food", r"اكل|غدا|عشا|فطار|فطرت|قهوه|كشري|فول|طعميه|بيتزا|برجر|شاورما|طلبات|talabat|عيش|خضار|فاكهه|سوبر ?ماركت|كارفور|food|lunch|dinner|coffee|grocer|koshary|عصير|شاي|حلويات"),
    ("Transport", r"اوبر|uber|كريم|careem|تاكسي|مترو|ميكروباص|بنزين|مواصلات|indrive|ديدي|didi|بنزينه|عربيه|قطر|اتوبيس|metro|taxi|fuel|bus|gas station"),
    ("Housing", r"ايجار|شقه|بواب|سباك|كهربائي|نقاش|rent|plumber"),
    ("Utilities", r"كهربا|كهرباء|ميه|مياه|غاز|نت|انترنت|فاتوره|رصيد|باقه|electric|water|internet|bill|recharge"),
    ("Subscriptions", r"اشتراك|نتفليكس|netflix|spotify|سبوتيفاي|شاهد|subscription|youtube premium|chatgpt"),
    ("Entertainment", r"سينما|خروجه|خرجت|ترفيه|ملاهي|بلايستيشن|cinema|movie|game|escape room|حفله"),
    ("Shopping", r"هدوم|لبس|كوتشي|جزمه|تيشيرت|بنطلون|موبايل|سماعه|لابتوب|هديه|امازون|amazon|نون|noon|جوميا|jumia|مول|shoes|clothes|gift"),
    ("Health", r"دكتور|كشف|دوا|صيدليه|تحاليل|جيم|gym|doctor|pharmacy|medicine|اسنان"),
    ("Education", r"كورس|كتب|كتاب|جامعه|مدرسه|دروس|مصاريف الدراسه|course|tuition|books|school"),
    ("Debt", r"قسط|سددت|دين|رجعت .*مستلف|installment|loan"),
    ("Salary", r"مرتب|راتب|بونص|salary|bonus|mortab"),
    ("Freelance", r"فريلانس|freelance|مشروع|عميل|client|خمسات|upwork"),
    ("Gift", r"عيديه|هديه|ادتني|ادالي|gift"),
    ("Refund", r"مرتجع|refund|رجعلي|استرجعت|استرداد"),
    ("Savings", r"توفير|شهاده|جمعيه|ادخار|savings"),
    ("Lending", r"سلفت|lent"),
    ("Transfer", r"حولت|تحويل|سحب|transfer|atm"),
]
ALLOWED = {
    "log_expense": {"Food", "Transport", "Housing", "Utilities", "Subscriptions", "Entertainment", "Shopping", "Health", "Education", "Debt", "Other"},
    "log_income": {"Salary", "Freelance", "Gift", "Refund", "Other"},
    "log_savings": {"Savings"},
    "log_transfer": {"Transfer", "Lending"},
}


class RulesApproach:
    name = "A. Rules (keywords + regex)"

    def train(self, recs):
        pass

    def size_bytes(self):
        return 0

    def predict(self, text):
        t = normalize(text)
        amount = extract_amount(text)
        intent = next((name for name, pat in INTENT_KEYWORDS if re.search(pat, t)), None)
        if intent is None:
            if GREETING.search(t):
                intent = "greeting"
            elif amount["amount"] is not None:
                intent = "log_expense"  # "netflix 240": a bare amount is most likely a purchase
            elif QUESTION.search(t):
                intent = "question"
            else:
                intent = "other"
        category = None
        if intent in ALLOWED:
            category = next((c for c, pat in CATEGORY_KEYWORDS if c in ALLOWED[intent] and re.search(pat, t)), None)
            if category is None:
                category = {"log_savings": "Savings", "log_transfer": "Transfer"}.get(intent, "Other")
        return {"intent": intent, "category": category, "category_conf": None, **amount, "entities": []}
