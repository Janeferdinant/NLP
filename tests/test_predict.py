import pytest

from spam_classifier.models import build_pipeline
from spam_classifier.predict import load_pipeline, predict_messages, save_pipeline


@pytest.fixture()
def fitted(toy_messages):
    texts, labels = toy_messages
    return build_pipeline("logreg", "word").fit(texts, labels)


def test_predict_messages_returns_label_and_score(fitted):
    out = predict_messages(fitted, ["WINNER!! Claim your free prize now", "Are we meeting for lunch today?"])
    assert [p.label for p in out] == ["spam", "ham"]
    assert 0.5 < out[0].score <= 1.0
    assert 0.0 <= out[1].score < 0.5


def test_threshold_moves_the_decision(fitted):
    msg = ["WINNER!! Claim your free prize now"]
    assert predict_messages(fitted, msg, threshold=0.0)[0].label == "spam"
    assert predict_messages(fitted, msg, threshold=1.01)[0].label == "ham"


def test_empty_input_gives_empty_output(fitted):
    assert predict_messages(fitted, []) == []


def test_save_and_load_round_trip(fitted, tmp_path):
    path = tmp_path / "sub" / "model.joblib"
    save_pipeline(fitted, path)
    again = load_pipeline(path)
    msgs = ["free prize call now", "see you at dinner"]
    assert [p.score for p in predict_messages(again, msgs)] == [p.score for p in predict_messages(fitted, msgs)]


def test_load_pipeline_missing_file_tells_the_user_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="spam_classifier train"):
        load_pipeline(tmp_path / "missing.joblib")
