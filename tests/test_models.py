import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from spam_classifier.models import MODEL_NAMES, build_pipeline


def test_model_names():
    assert MODEL_NAMES == ("nb", "logreg", "svm")


@pytest.mark.parametrize("name, cls", [("nb", MultinomialNB), ("logreg", LogisticRegression), ("svm", LinearSVC)])
def test_pipeline_has_a_vectorizer_and_the_right_classifier(name, cls):
    pipe = build_pipeline(name, "word")
    assert list(pipe.named_steps) == ["tfidf", "clf"]
    assert isinstance(pipe.named_steps["clf"], cls)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("features", ["word", "char", "word+char"])
def test_every_combination_fits_and_separates_the_toy_corpus(name, features, toy_messages):
    texts, labels = toy_messages
    pipe = build_pipeline(name, features).fit(texts, labels)
    assert list(pipe.predict(texts)) == labels


def test_nb_defaults_match_the_notebook():
    clf = build_pipeline("nb", "word").named_steps["clf"]
    assert clf.alpha == 1.0 and clf.fit_prior is True


def test_class_weight_is_passed_to_linear_models():
    assert build_pipeline("logreg", "word", class_weight="balanced").named_steps["clf"].class_weight == "balanced"
    assert build_pipeline("svm", "word", class_weight="balanced").named_steps["clf"].class_weight == "balanced"


def test_class_weight_is_rejected_for_naive_bayes():
    with pytest.raises(ValueError, match="class_weight"):
        build_pipeline("nb", "word", class_weight="balanced")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="forest"):
        build_pipeline("forest", "word")
