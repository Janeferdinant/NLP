"""Load the UCI SMS Spam Collection and split it exactly as the notebook does."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0
TEST_SIZE = 0.3
ENCODING = "ISO-8859-1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "spam.csv"


def load_dataset(path=None):
    """Read the CSV into a frame with columns ``class_label`` (1 = spam, 0 = ham) and ``message``.

    The file is not valid UTF-8, so it is read as ISO-8859-1, as in the notebook. Only the first two
    columns (``v1`` label, ``v2`` text) are used; the three unnamed trailing columns are ignored.
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, encoding=ENCODING, usecols=[0, 1])
    df.columns = ["class_label", "message"]
    df["class_label"] = df["class_label"].apply(lambda x: 1 if x == "spam" else 0)
    return df


def split_dataset(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Return ``x_train, x_test, y_train, y_test`` using the notebook's split (no stratification)."""
    return train_test_split(df["message"], df["class_label"], test_size=test_size, random_state=random_state)
