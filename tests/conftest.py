"""Shared fixtures: a tiny synthetic corpus and a tiny CSV in the UCI file format."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = REPO_ROOT / "data" / "spam.csv"

SPAM_TEMPLATES = [
    "WINNER!! Claim your free prize now, call 0{n} to collect",
    "URGENT! You have won a guaranteed cash award, txt WIN to {n}",
    "Free entry in our weekly competition, text STOP to {n} to opt out",
    "Congratulations, you have been selected for a free holiday, click http://win{n}.example",
]
HAM_TEMPLATES = [
    "Are we still meeting for lunch at {n} today?",
    "I will be home late, please save me some dinner",
    "Can you pick up the kids from school at {n}, thanks",
    "Sorry I missed your call, talk to you tomorrow",
]


@pytest.fixture(scope="session")
def toy_messages():
    """40 messages (20 spam, 20 ham) and their 0/1 labels."""
    texts, labels = [], []
    for n in range(20):
        texts.append(SPAM_TEMPLATES[n % 4].format(n=80000 + n))
        labels.append(1)
        texts.append(HAM_TEMPLATES[n % 4].format(n=n % 12 + 1))
        labels.append(0)
    return texts, labels


@pytest.fixture()
def toy_csv(tmp_path, toy_messages):
    """The toy corpus written in the UCI layout (v1, v2 and three empty columns), ISO-8859-1."""
    texts, labels = toy_messages
    lines = ["v1,v2,,,"]
    for text, label in zip(texts, labels):
        lines.append(f'{"spam" if label else "ham"},"{text}",,,')
    path = tmp_path / "toy.csv"
    path.write_bytes("\n".join(lines).encode("ISO-8859-1"))
    return path


needs_real_data = pytest.mark.skipif(not REAL_DATA.exists(), reason="data/spam.csv is not present")
