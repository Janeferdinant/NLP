import numpy as np
import pytest

from spam_classifier.evaluation import (
    bootstrap_accuracy_ci,
    compute_metrics,
    error_cases,
    operating_point,
    out_of_fold_scores,
    spam_scores,
    threshold_for_precision,
    threshold_sweep,
    top_terms,
)
from spam_classifier.models import build_pipeline


def test_compute_metrics_on_a_known_confusion_matrix():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["tn"] == 3 and m["fp"] == 1 and m["fn"] == 2 and m["tp"] == 2
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_operating_point_applies_the_threshold_to_the_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    at_half = operating_point(y, scores, 0.5)
    assert (at_half["tp"], at_half["fp"], at_half["fn"], at_half["tn"]) == (1, 1, 1, 1)
    at_low = operating_point(y, scores, 0.3)
    assert (at_low["tp"], at_low["fp"], at_low["fn"], at_low["tn"]) == (2, 1, 0, 1)
    assert at_low["threshold"] == 0.3


def test_threshold_sweep_recall_never_increases_as_the_threshold_rises():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    rows = threshold_sweep(y, scores, [0.1, 0.3, 0.5, 0.7, 0.9])
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert [r["threshold"] for r in rows] == [0.1, 0.3, 0.5, 0.7, 0.9]


def test_precision_is_one_when_nothing_is_flagged():
    row = operating_point(np.array([0, 1]), np.array([0.1, 0.2]), 0.9)
    assert row["tp"] == 0 and row["fp"] == 0
    assert row["precision"] == 1.0 and row["recall"] == 0.0


def test_threshold_for_precision_picks_the_lowest_threshold_meeting_the_target():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.55, 0.5, 0.6, 0.8, 0.9])
    # At 0.6 or above: 3 spam, 0 ham flagged. At 0.5: 4 spam + 1 ham (precision 0.8).
    assert threshold_for_precision(y, scores, 1.0) == pytest.approx(0.6)
    assert threshold_for_precision(y, scores, 0.8) == pytest.approx(0.5)


def test_threshold_for_precision_raises_when_unreachable():
    y = np.array([0, 1])
    scores = np.array([0.9, 0.1])
    with pytest.raises(ValueError):
        threshold_for_precision(y, scores, 1.0)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.9, y, 1 - y)
    low, high = bootstrap_accuracy_ci(y, pred, n_resamples=500, seed=0)
    acc = float(np.mean(y == pred))
    assert low <= acc <= high
    assert (low, high) == bootstrap_accuracy_ci(y, pred, n_resamples=500, seed=0)


def test_spam_scores_are_probabilities_for_nb_and_logreg_and_margins_for_svm(toy_messages):
    texts, labels = toy_messages
    for name in ("nb", "logreg"):
        s = spam_scores(build_pipeline(name, "word").fit(texts, labels), texts)
        assert s.min() >= 0.0 and s.max() <= 1.0
    s = spam_scores(build_pipeline("svm", "word").fit(texts, labels), texts)
    assert s.min() < 0.0 < s.max()


def test_out_of_fold_scores_cover_every_row_without_fitting_the_given_pipeline(toy_messages):
    texts, labels = toy_messages
    pipe = build_pipeline("logreg", "word")
    scores = out_of_fold_scores(pipe, texts, labels, n_splits=4, seed=0)
    assert scores.shape == (40,)
    assert not hasattr(pipe.named_steps["clf"], "coef_")


def test_top_terms_returns_spam_and_ham_words(toy_messages):
    texts, labels = toy_messages
    for name in ("nb", "logreg", "svm"):
        pipe = build_pipeline(name, "word").fit(texts, labels)
        spam_terms, ham_terms = top_terms(pipe, 5)
        assert len(spam_terms) == 5 and len(ham_terms) == 5
        assert {"free", "prize", "win", "claim"} & {t for t, _ in spam_terms}
        assert {"dinner", "lunch", "school", "tomorrow"} & {t for t, _ in ham_terms}


def test_error_cases_split_false_negatives_and_false_positives():
    texts = ["a", "b", "c", "d"]
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    scores = [0.9, 0.2, 0.7, 0.1]
    fn, fp = error_cases(texts, y_true, y_pred, scores)
    assert [e["text"] for e in fn] == ["b"]
    assert [e["text"] for e in fp] == ["c"]
    assert fn[0]["score"] == pytest.approx(0.2)
