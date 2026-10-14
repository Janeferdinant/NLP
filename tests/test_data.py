import pandas as pd
import pytest

from spam_classifier.data import (
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    load_dataset,
    split_dataset,
)
from tests.conftest import needs_real_data


def test_split_constants_match_the_notebook():
    assert RANDOM_STATE == 0
    assert TEST_SIZE == 0.3


def test_load_dataset_uses_notebook_column_names_and_numeric_labels(toy_csv):
    df = load_dataset(toy_csv)
    assert list(df.columns) == ["class_label", "message"]
    assert set(df["class_label"]) == {0, 1}
    assert int(df["class_label"].sum()) == 20
    assert len(df) == 40


def test_load_dataset_reads_iso_8859_1(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("v1,v2,,,\nham,caf\xe9 at 5,,,\nspam,win \xa3100 now,,,\n".encode("ISO-8859-1"))
    df = load_dataset(path)
    assert df["message"].tolist() == ["caf\xe9 at 5", "win \xa3100 now"]


def test_load_dataset_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_dataset(tmp_path / "nope.csv")


def test_split_dataset_is_deterministic_and_sized(toy_csv):
    df = load_dataset(toy_csv)
    a = split_dataset(df)
    b = split_dataset(df)
    x_train, x_test, y_train, y_test = a
    assert len(x_train) == 28 and len(x_test) == 12
    assert len(y_train) == 28 and len(y_test) == 12
    pd.testing.assert_series_equal(a[1], b[1])


@needs_real_data
def test_real_dataset_counts_and_split_match_the_notebook():
    df = load_dataset(DEFAULT_DATA_PATH)
    assert len(df) == 5572
    assert int(df["class_label"].sum()) == 747
    x_train, x_test, y_train, y_test = split_dataset(df)
    assert (len(x_train), len(x_test)) == (3900, 1672)
    assert int(y_test.sum()) == 238
    assert int((y_test == 0).sum()) == 1434
