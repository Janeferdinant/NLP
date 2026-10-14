"""TF-IDF feature builders: word unigrams, character n-grams, or both."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .preprocessing import normalize_text

FEATURE_NAMES = ("word", "char", "word+char")


def _word(normalize):
    # The notebook's configuration: lowercase, English stop words, unigrams, default token pattern.
    if normalize:
        return TfidfVectorizer(preprocessor=normalize_text, stop_words="english")
    return TfidfVectorizer(lowercase=True, stop_words="english")


def _char(normalize):
    # Character n-grams inside word boundaries, 2 to 5 characters. No stop-word list: it makes no sense
    # for characters. Sub-linear term frequency and min_df=2 keep the feature space manageable.
    kwargs = {"preprocessor": normalize_text} if normalize else {"lowercase": True}
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, **kwargs)


def build_vectorizer(kind, normalize=False):
    """Return an unfitted vectorizer. ``kind`` is one of ``FEATURE_NAMES``."""
    if kind == "word":
        return _word(normalize)
    if kind == "char":
        return _char(normalize)
    if kind == "word+char":
        return FeatureUnion([("word", _word(normalize)), ("char", _char(normalize))])
    raise ValueError(f"Unknown feature kind {kind!r}; choose one of {', '.join(FEATURE_NAMES)}")
