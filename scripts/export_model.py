"""Export the trained TF-IDF vectorizer and Naive Bayes model to JSON for the in-browser demo.

Run this after executing spam_ham.ipynb. The notebook's last cell saves
model/vectorizer.joblib and model/nb.joblib.

    python scripts/export_model.py

Writes:
  docs/model.json     vocabulary, idf, stop words, vectorizer params, classes,
                      class_log_prior and feature_log_prob
  docs/fixtures.json  the 1,672 held-out test messages with scikit-learn's
                      predictions; docs/parity.test.mjs checks the JavaScript
                      classifier against them
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DOCS = ROOT / "docs"


def load_test_split():
    """Load spam.csv and split it exactly as spam_ham.ipynb does (cells 1, 12 and 14)."""
    df = pd.read_csv(ROOT / "spam.csv", encoding="ISO-8859-1")
    df = df.rename(columns={"v1": "class_label", "v2": "message"})[["class_label", "message"]]
    df["class_label"] = df["class_label"].apply(lambda x: 1 if x == "spam" else 0)
    _, x_test, _, y_test = train_test_split(
        df["message"], df["class_label"], test_size=0.3, random_state=0
    )
    return x_test, y_test


def export_model(vec, clf):
    p = vec.get_params()
    return {
        "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "stop_words": sorted(vec.get_stop_words() or []),
        "params": {k: p[k] for k in ("lowercase", "ngram_range", "sublinear_tf", "norm", "token_pattern")},
        "classes": clf.classes_.tolist(),
        "class_log_prior": clf.class_log_prior_.tolist(),
        "feature_log_prob": clf.feature_log_prob_.tolist(),
    }


def main():
    vec = joblib.load(MODEL_DIR / "vectorizer.joblib")
    clf = joblib.load(MODEL_DIR / "nb.joblib")

    DOCS.mkdir(exist_ok=True)
    model_path = DOCS / "model.json"
    with open(model_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(export_model(vec, clf), f, separators=(",", ":"))
        f.write("\n")

    x_test, y_test = load_test_split()
    pred = clf.predict(vec.transform(x_test))
    fixtures_path = DOCS / "fixtures.json"
    rows = [
        json.dumps({"text": text, "label": int(label), "sklearn": int(p)})
        for text, label, p in zip(x_test, y_test, pred)
    ]
    with open(fixtures_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("[\n" + ",\n".join(rows) + "\n]\n")

    n = len(y_test)
    print(f"vocabulary size: {len(vec.vocabulary_)}")
    print(f"wrote {model_path.relative_to(ROOT)} ({model_path.stat().st_size:,} bytes)")
    print(f"wrote {fixtures_path.relative_to(ROOT)} ({fixtures_path.stat().st_size:,} bytes, {n} messages)")
    print(f"test messages: {n} (spam: {int(y_test.sum())}, ham: {int((y_test == 0).sum())})")
    print(f"confusion matrix [[TN, FP], [FN, TP]]: {confusion_matrix(y_test, pred).tolist()}")
    print(f"accuracy:            {accuracy_score(y_test, pred):.6f}")
    print(f"precision (spam):    {precision_score(y_test, pred):.6f}")
    print(f"recall (spam):       {recall_score(y_test, pred):.6f}")
    print(f"F1 (spam):           {f1_score(y_test, pred):.6f}")
    print(f"majority baseline:   {accuracy_score(y_test, [0] * n):.6f} (always predict ham)")


if __name__ == "__main__":
    main()
