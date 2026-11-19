"""Text normalization shared by every approach."""
import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
_DIACRITICS = re.compile(r"[ً-ْـ]")  # harakat + tatweel


def to_western_digits(text: str) -> str:
    return text.translate(_AR_DIGITS)


def normalize_letters(text: str) -> str:
    """Unify Arabic letter variants so keyword and word-number matching is robust."""
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def normalize(text: str) -> str:
    return normalize_letters(to_western_digits(text)).lower()
