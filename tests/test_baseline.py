"""The 96.41% Naive Bayes result from the notebook must be reproducible from the package."""
import pytest
from sklearn.metrics import confusion_matrix

from spam_classifier.data import DEFAULT_DATA_PATH, load_dataset, split_dataset
from spam_classifier.models import build_pipeline
from tests.conftest import needs_real_data


@needs_real_data
def test_nb_word_tfidf_reproduces_the_notebook_numbers():
    df = load_dataset(DEFAULT_DATA_PATH)
    x_train, x_test, y_train, y_test = split_dataset(df)
    pipe = build_pipeline("nb", "word").fit(x_train.tolist(), y_train)
    pred = pipe.predict(x_test)
    assert pipe.score(x_test, y_test) == pytest.approx(0.9641148325358851, abs=1e-12)
    assert confusion_matrix(y_test, pred).tolist() == [[1434, 0], [60, 178]]
