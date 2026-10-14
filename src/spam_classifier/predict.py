"""Save, load and apply a trained pipeline."""

from pathlib import Path
from typing import NamedTuple

import joblib

from .data import REPO_ROOT
from .evaluation import default_threshold, spam_scores

DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "spam_classifier.joblib"


class Prediction(NamedTuple):
    label: str  # "spam" or "ham"
    score: float  # probability of spam (Naive Bayes, logistic regression) or signed margin (SVM)


def save_pipeline(pipeline, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path=None):
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m spam_classifier train` first.")
    return joblib.load(path)


def predict_messages(pipeline, messages, threshold=None):
    """Classify messages. A message is spam when its score is at least ``threshold``."""
    messages = list(messages)
    if not messages:
        return []
    if threshold is None:
        threshold = default_threshold(pipeline)
    scores = spam_scores(pipeline, messages)
    return [Prediction("spam" if s >= threshold else "ham", float(s)) for s in scores]
