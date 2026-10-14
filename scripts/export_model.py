"""Export the Naive Bayes baseline (TF-IDF vectorizer and classifier) to JSON for the in-browser demo.

The demo runs the notebook's model: word TF-IDF with English stop words and Multinomial Naive Bayes, trained
on the notebook's 70% split. This script trains that model with the spam_classifier package and writes:

    python scripts/export_model.py

  docs/model.json     vocabulary, idf, stop words, vectorizer params, classes,
                      class_log_prior and feature_log_prob
  docs/fixtures.json  the 1,672 held-out test messages with scikit-learn's
                      predictions; docs/parity.test.mjs checks the JavaScript
                      classifier against them
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from spam_classifier.data import load_dataset, split_dataset
from spam_classifier.models import build_pipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


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
    x_train, x_test, y_train, y_test = split_dataset(load_dataset())
    pipe = build_pipeline("nb", "word").fit(x_train.tolist(), y_train)
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]

    DOCS.mkdir(exist_ok=True)
    model_path = DOCS / "model.json"
    with open(model_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(export_model(vec, clf), f, separators=(",", ":"))
        f.write("\n")

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
