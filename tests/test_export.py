"""The exported JSON must let another implementation reproduce the scikit-learn pipelines."""

import numpy as np
import pytest

from spam_classifier.export import analyzer_fixtures, build_fixtures, export_nb, export_svm
from spam_classifier.models import build_pipeline


@pytest.fixture(scope="module")
def svm(toy_messages):
    texts, labels = toy_messages
    return build_pipeline("svm", "word+char").fit(texts, labels)


@pytest.fixture(scope="module")
def nb(toy_messages):
    texts, labels = toy_messages
    return build_pipeline("nb", "word").fit(texts, labels)


def test_export_svm_schema(svm):
    out = export_svm(svm, n_docs=40)
    assert out["model"] == "svm"
    assert out["classes"] == [0, 1]
    assert set(out["word"]) == {"params", "stop_words", "terms", "df", "n_docs"}
    assert set(out["char"]) == {"params", "terms", "df", "n_docs"}
    assert out["char"]["params"]["analyzer"] == "char_wb"
    assert out["char"]["params"]["ngram_range"] == [2, 5]
    assert out["char"]["params"]["sublinear_tf"] is True
    assert out["word"]["params"]["lowercase"] is True
    assert len(out["coef"]) == len(out["word"]["terms"]) + len(out["char"]["terms"])
    assert all(isinstance(d, int) for d in out["word"]["df"] + out["char"]["df"])


def test_export_svm_document_frequencies_rebuild_the_idf_exactly(svm):
    out = export_svm(svm, n_docs=40)
    for name, vec in zip(("word", "char"), (t for _, t in svm.named_steps["tfidf"].transformer_list), strict=True):
        part = out[name]
        idf = np.log((1 + part["n_docs"]) / (1 + np.array(part["df"]))) + 1
        np.testing.assert_allclose(idf, vec.idf_, rtol=0, atol=1e-12)
        assert part["terms"] == sorted(vec.vocabulary_, key=vec.vocabulary_.get)


def test_export_svm_coefficients_reproduce_the_decision_function(svm, toy_messages):
    texts, _ = toy_messages
    out = export_svm(svm, n_docs=40)
    x = svm.named_steps["tfidf"].transform(texts)
    scores = x @ np.array(out["coef"]) + out["intercept"]
    np.testing.assert_allclose(scores, svm.decision_function(texts), atol=1e-5)


def test_export_nb_keeps_the_existing_schema(nb):
    out = export_nb(nb.named_steps["tfidf"], nb.named_steps["clf"])
    assert set(out) == {
        "vocabulary",
        "idf",
        "stop_words",
        "params",
        "classes",
        "class_log_prior",
        "feature_log_prob",
    }


def test_analyzer_fixtures_cover_awkward_inputs_and_match_sklearn(svm):
    cases = analyzer_fixtures(svm, ["a free prize"])
    texts = [c["text"] for c in cases]
    char = dict(svm.named_steps["tfidf"].transformer_list)["char"]
    assert "a free prize" in texts
    assert any("\x85" in t for t in texts), "a NEL (U+0085) case is needed: Python treats it as whitespace"
    assert any("\U0001f600" in t for t in texts), "a non-BMP character case is needed"
    assert any("  " in t for t in texts) and any("\t" in t for t in texts)
    assert any(t == "I" for t in texts), "a one-character word case is needed"
    for case in cases:
        assert case["ngrams"] == char.build_analyzer()(case["text"])


def test_build_fixtures_hold_both_models_predictions(nb, svm, toy_messages):
    texts, labels = toy_messages
    rows = build_fixtures(texts, labels, nb, svm)
    assert len(rows) == len(texts)
    row = rows[0]
    assert set(row) == {"text", "label", "sklearn", "svm", "svm_score"}
    assert row["sklearn"] == int(nb.predict([texts[0]])[0])
    assert row["svm"] == int(svm.predict([texts[0]])[0])
    assert row["svm_score"] == pytest.approx(float(svm.decision_function([texts[0]])[0]), abs=1e-9)
