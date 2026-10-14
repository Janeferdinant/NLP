"""Optional text normalisation.

The baseline from the notebook does not use this module: it lets ``TfidfVectorizer`` lowercase the text
and remove English stop words. ``normalize_text`` is an experiment (``--normalize``) that replaces the
parts of a message that vary from spam to spam (links, prices, phone numbers) with fixed tokens, so
that the model can learn "has a link" instead of one specific link.
"""
import re

_URL = re.compile(r"(?:https?://|www\.)\S+")
_MONEY = re.compile(r"[\xa3$€]\s?\d[\d,.]*")
_PHONE = re.compile(r"\d{7,}")
_SPACE = re.compile(r"\s+")


def normalize_text(text):
    """Lowercase, replace URLs, money amounts and long digit runs by tokens, and collapse whitespace."""
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    return _SPACE.sub(" ", text).strip()
