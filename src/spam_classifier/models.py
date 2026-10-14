"""The three classifier families, each wrapped with its vectorizer in one scikit-learn pipeline."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_vectorizer

MODEL_NAMES = ("nb", "logreg", "svm")
# The best model of the 2026-10-14 comparison by cross-validated F1 on the training split. The notebook's baseline is
# `--algorithm nb --features word`.
DEFAULT_ALGORITHM = "svm"
DEFAULT_FEATURES = "word+char"
DISPLAY_NAMES = {"nb": "Multinomial Naive Bayes", "logreg": "Logistic regression", "svm": "Linear SVM"}


def build_classifier(name, class_weight=None, **params):
    if name == "nb":
        if class_weight is not None:
            raise ValueError("class_weight is not supported by Naive Bayes")
        return MultinomialNB(**params)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=0, class_weight=class_weight, **params)
    if name == "svm":
        return LinearSVC(random_state=0, class_weight=class_weight, **params)
    raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(MODEL_NAMES)}")


def build_pipeline(name, features="word", normalize=False, class_weight=None, **params):
    """Return an unfitted ``Pipeline([("tfidf", ...), ("clf", ...)])``."""
    clf = build_classifier(name, class_weight=class_weight, **params)
    return Pipeline([("tfidf", build_vectorizer(features, normalize=normalize)), ("clf", clf)])
