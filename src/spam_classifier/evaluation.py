"""Metrics, decision-threshold analysis, top terms and error cases.

Conventions: spam is the positive class (label 1). A score is "spam-ness": a probability in [0, 1] for
Naive Bayes and logistic regression, a signed margin for the linear SVM. A message is flagged as spam when
``score >= threshold``. When nothing is flagged, precision is reported as 1.0 (no false alarms), which is
the convention of ``sklearn.metrics.precision_recall_curve``.
"""
import re

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def _rates(tn, fp, fn, tp):
    flagged = tp + fp
    positives = tp + fn
    precision = tp / flagged if flagged else 1.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tn + fp + fn + tp
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": (tn + tp) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(y_true, y_pred):
    """Confusion-matrix counts plus accuracy, precision, recall and F1 for spam."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return _rates(tn, fp, fn, tp)


def spam_scores(pipeline, texts):
    """Probability of spam when the classifier has one, otherwise the signed SVM margin."""
    texts = list(texts)
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)


def default_threshold(pipeline):
    """The threshold at which ``predict`` flips: 0.5 for probabilities, 0.0 for margins."""
    return 0.5 if hasattr(pipeline, "predict_proba") else 0.0


def operating_point(y_true, scores, threshold):
    """Metrics when every message with ``score >= threshold`` is flagged as spam."""
    row = compute_metrics(y_true, (np.asarray(scores) >= threshold).astype(int))
    row["threshold"] = threshold
    return row


def threshold_sweep(y_true, scores, thresholds):
    return [operating_point(y_true, scores, t) for t in thresholds]


def threshold_for_precision(y_true, scores, min_precision):
    """The lowest threshold (so the highest recall) whose precision is at least ``min_precision``.

    Raises ``ValueError`` when no threshold that flags at least one spam message reaches the target.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    tp = np.cumsum(y_true[order])
    precision = tp / np.arange(1, len(s) + 1)
    group_end = np.append(s[1:] != s[:-1], True)  # last item of each run of equal scores
    ok = np.nonzero(group_end & (precision >= min_precision) & (tp > 0))[0]
    if ok.size == 0:
        raise ValueError(f"No threshold reaches precision {min_precision}")
    return float(s[ok[-1]])


def bootstrap_accuracy_ci(y_true, y_pred, n_resamples=1000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for accuracy, resampling test messages with replacement."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
    accs = correct[idx].mean(axis=1)
    low, high = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def out_of_fold_scores(pipeline, texts, y, n_splits=5, seed=0):
    """Scores for every message from models that never saw that message (stratified k-fold)."""
    texts = np.asarray(list(texts), dtype=object)
    y = np.asarray(y).astype(int)
    scores = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, held_idx in folds.split(texts, y):
        model = clone(pipeline).fit(list(texts[train_idx]), y[train_idx])
        scores[held_idx] = spam_scores(model, texts[held_idx])
    return scores


def top_terms(pipeline, n=20):
    """The ``n`` strongest spam terms and ``n`` strongest ham terms as ``(term, weight)`` pairs.

    The weight is the log-probability ratio spam/ham for Naive Bayes and the coefficient for the linear
    models. Spam terms have positive weights (largest first); ham terms negative (most negative first).
    """
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        weights = clf.coef_[0]
    order = np.argsort(weights)
    spam = [(str(names[i]), float(weights[i])) for i in order[::-1][:n]]
    ham = [(str(names[i]), float(weights[i])) for i in order[:n]]
    return spam, ham


def error_cases(texts, y_true, y_pred, scores):
    """Split the mistakes into false negatives (missed spam) and false positives (flagged ham).

    False negatives are ordered from the lowest score (most confidently missed); false positives from the
    highest score.
    """
    fn, fp = [], []
    for text, truth, pred, score in zip(texts, y_true, y_pred, scores, strict=True):
        if truth == 1 and pred == 0:
            fn.append({"text": text, "score": float(score)})
        elif truth == 0 and pred == 1:
            fp.append({"text": text, "score": float(score)})
    fn.sort(key=lambda e: e["score"])
    fp.sort(key=lambda e: -e["score"])
    return fn, fp


_LONG_NUMBER = re.compile(r"(?<!\d)\+?\d(?:[ -]?\d){6,}(?!\d)")


def mask_digits(text):
    """Replace phone-number-like digit runs (7 or more digits) by ``[number]`` before text is quoted."""
    return _LONG_NUMBER.sub("[number]", text)
