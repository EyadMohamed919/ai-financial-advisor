"""Item type (sub-category) lookup, e.g. "لاب" -> Electronics, "كوتشي" -> Shoes.

The training data has no item-type labels, so this is a keyword lexicon for now.
Once real data is labeled with item types, it can be replaced by a classifier.
"""
import re

from .normalize import normalize

# type: (Arabic label, keywords in normalized form: أ/إ/آ -> ا, ة -> ه, ى -> ي)
ITEM_TYPES = {
    "Electronics": ("إلكترونيات", "لاب لابتوب laptop موبايل تليفون فون ايفون iphone سامسونج samsung شاومي سماعه سماعات ايربودز airpods "
                    "شاحن تابلت ايباد ipad شاشه تلفزيون تليفزيون كيبورد ماوس بلايستيشن playstation ps5 كاميرا ساعه سمارت باور بانك"),
    "Clothes": ("ملابس", "تيشيرت تيشرت قميص بنطلون جينز جاكيت بلوزه فستان هدوم لبس عبايه شورت هودي hoodie بدله جيبه شراب بيجامه"),
    "Shoes": ("أحذية", "كوتشي كوتشيات جزمه صندل شبشب sneakers shoes"),
    "Groceries": ("بقالة", "عيش لبن جبنه خضار فاكهه رز سكر زيت لحمه فراخ بيض مكرونه تموين بقاله سمك groceries"),
    "Meals": ("أكل جاهز", "بيتزا برجر كشري شاورما فول طعميه سندوتش سندوتشات وجبه كريب فطير pizza burger koshary"),
    "Drinks": ("مشروبات وقهوة", "قهوه شاي لاتيه نسكافيه عصير كابتشينو coffee latte"),
    "Fuel": ("بنزين وشحن", "بنزين سولار بنزينه شحن العربيه fuel"),
    "Rides": ("مشاوير", "اوبر uber كريم careem تاكسي ميكروباص مترو اتوبيس توكتوك مشوار indrive"),
    "Medicine": ("أدوية", "دوا ادويه علاج فيتامين روشته"),
    "Medical": ("كشف وتحاليل", "كشف تحاليل اشعه دكتور اسنان"),
    "Books & courses": ("كتب وكورسات", "كتاب كتب كورس كورسات ملزمه دروس"),
    "Appliances": ("أجهزة منزلية", "تلاجه غساله بوتاجاز مكيف تكييف ميكروويف خلاط مكنسه سخان"),
    "Furniture": ("أثاث", "كنبه سرير ترابيزه دولاب كرسي مكتب"),
    "Personal care": ("عناية شخصية", "برفان ميكب شامبو كريم حلاق كوافير"),
    "Gifts": ("هدايا", "هديه هدايا ورد بوكيه"),
    "Mobile credit": ("رصيد وباقات", "رصيد باقه كارت فكه"),
    "Bills": ("فواتير", "فاتوره كهربا كهرباء مياه ميه غاز انترنت نت"),
}

_PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب")
_LOOKUP = {kw: t for t, (_, words) in ITEM_TYPES.items() for kw in words.split() if " " not in kw}
_PHRASES = {"شحن العربيه": "Fuel", "باور بانك": "Electronics", "ساعه سمارت": "Electronics"}


def _variants(token):
    yield token
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 2:
            yield token[len(p):]


def item_type(item, text):
    """Return (type, Arabic label) from the item, falling back to the whole message; (None, None) if unknown."""
    for source in (item, text):
        if not source:
            continue
        t = normalize(source)
        for phrase, typ in _PHRASES.items():
            if phrase in t:
                return typ, ITEM_TYPES[typ][0]
        for token in re.findall(r"[\w]+", t):
            for v in _variants(token):
                if v in _LOOKUP:
                    typ = _LOOKUP[v]
                    return typ, ITEM_TYPES[typ][0]
    return None, None


# "اشتريت لاب من ..." -> "لاب": what follows a purchase verb, up to a place, amount or date word.
ITEM_RE = re.compile(
    r"(?:اشتريت|اشترينا|جبت|طلبت|خدت|bought|got|ordered)\s+(.+?)"
    r"(?=\s+(?:من|عند|في|ب|بـ|بمبلغ|ب?\d|امبارح|النهارده|النهاردة|from|at|for|on)\b|\s*[\d٠-٩,.،!؟?]|\s*$)",
    re.I)


def rule_item(text):
    m = ITEM_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    return name if 0 < len(name.split()) <= 4 else None
