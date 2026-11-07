"""Text cleaning strategies for SMS spam detection.

Two variants are compared:

minimal
    Lowercase only. The vectorizer handles tokenisation via its built-in
    token pattern. Preserves all original characters so the model can learn
    from punctuation and digit patterns natively.

heavy
    Lowercase → replace digit runs with TOKEN_NUM (generalises "win £150"
    and "win £200" to the same token) → keep £, $, €, ! and ? as explicit
    spam signals while removing all other punctuation → strip common English
    stopwords → apply Porter stemming. The stemmer reduces "winning",
    "winner", "wins" to a common root so sparse spam vocabulary is denser.

Why keep currency symbols and !, ?
    Real spam frequently uses "FREE!", "£150 prize", "txt STOP" - these are
    strong discriminative signals that disappear under aggressive punctuation
    removal. We therefore only strip punctuation that carries no signal
    (commas, brackets, etc.) while retaining the high-value characters.

Why digit normalisation instead of full removal
    Digit patterns ("150", "08001234567") are common in spam but removing
    them entirely loses the signal. Replacing any digit sequence with the
    single token NUM preserves the "there is a number here" signal while
    reducing sparsity.
"""
import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data once at import time (silent if already present)
for _resource in ("stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_resource}" if _resource == "stopwords"
                       else f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOPWORDS  = set(stopwords.words("english"))
_STEMMER    = PorterStemmer()
# Punctuation to keep as spam signals
_KEEP_PUNCT = set("£$€!?")
_STRIP_PUNCT = "".join(c for c in string.punctuation if c not in _KEEP_PUNCT)
_DIGIT_RE   = re.compile(r"\d+")


def clean_minimal(text: str) -> str:
    """Lowercase only. The vectorizer handles the rest."""
    return text.lower()


def clean_heavy(text: str) -> str:
    """Aggressive cleaning: normalise digits, strip non-signal punctuation,
    remove stopwords, apply Porter stemming."""
    text = text.lower()
    text = _DIGIT_RE.sub(" NUM ", text)
    text = text.translate(str.maketrans("", "", _STRIP_PUNCT))
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


CLEANERS = {
    "minimal": clean_minimal,
    "heavy":   clean_heavy,
}
