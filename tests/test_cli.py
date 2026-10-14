import pytest

from spam_classifier.cli import main
from tests.conftest import needs_real_data


def test_train_then_predict_round_trip(toy_csv, tmp_path, capsys):
    model = tmp_path / "m.joblib"
    assert main(["train", "--data", str(toy_csv), "--output", str(model), "--algorithm", "logreg"]) == 0
    assert model.exists()
    assert "accuracy" in capsys.readouterr().out
    assert main(["predict", "--model-path", str(model), "WINNER!! Claim your free prize now", "lunch at 5?"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("spam")
    assert out[1].startswith("ham")


def test_predict_without_a_trained_model_fails_with_instructions(tmp_path, capsys):
    code = main(["predict", "--model-path", str(tmp_path / "none.joblib"), "hello"])
    assert code != 0
    assert "spam_classifier train" in capsys.readouterr().err


def test_predict_reads_messages_from_a_file(toy_csv, tmp_path, capsys):
    model = tmp_path / "m.joblib"
    main(["train", "--data", str(toy_csv), "--output", str(model)])
    capsys.readouterr()
    messages = tmp_path / "msgs.txt"
    messages.write_text("free prize claim now\nsee you at dinner\n", encoding="utf-8")
    assert main(["predict", "--model-path", str(model), "--file", str(messages)]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_evaluate_reports_the_saved_models_test_metrics(toy_csv, tmp_path, capsys):
    model = tmp_path / "m.joblib"
    main(["train", "--data", str(toy_csv), "--output", str(model)])
    capsys.readouterr()
    assert main(["evaluate", "--data", str(toy_csv), "--model-path", str(model)]) == 0
    out = capsys.readouterr().out
    assert "precision" in out and "recall" in out and "confusion matrix" in out


def test_unknown_algorithm_is_an_argparse_error(toy_csv):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--data", str(toy_csv), "--algorithm", "forest"])
    assert exc.value.code == 2


@needs_real_data
def test_default_training_reproduces_the_notebook_result(tmp_path, capsys):
    """Naive Bayes + word TF-IDF on the notebook's split: 96.41% accuracy, 0 false positives, 60 false negatives."""
    model = tmp_path / "nb.joblib"
    assert main(["train", "--algorithm", "nb", "--features", "word", "--output", str(model)]) == 0
    out = capsys.readouterr().out
    assert "accuracy 0.964115" in out
    assert "[[1434, 0], [60, 178]]" in out
