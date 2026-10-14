"""Export the demo's models and test fixtures.

    python scripts/export_model.py

The demo offers two models, both trained with the spam_classifier package on the notebook's 70% split:

* the notebook's Naive Bayes (word TF-IDF)                      -> docs/model.json
* the best model, a linear SVM on word + character TF-IDF       -> docs/model-svm.json

and the fixtures that docs/parity.test.mjs and docs/svm.test.mjs check the JavaScript against:

  docs/fixtures.json           the 1,672 held-out test messages with scikit-learn's Naive Bayes and SVM
                               predictions and the SVM decision values
  docs/fixtures-analyzer.json  scikit-learn's character n-grams for awkward inputs and 40 real messages
"""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from spam_classifier.data import load_dataset, split_dataset
from spam_classifier.export import analyzer_fixtures, build_fixtures, export_nb, export_svm
from spam_classifier.models import build_pipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    return path


def write_rows(path, rows):
    """One JSON object per line, so diffs stay readable."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
    return path


def size(path):
    return f"{path.stat().st_size:,} bytes"


def report(name, y_test, pred):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"{name}: confusion matrix [[TN, FP], [FN, TP]]: {[[int(tn), int(fp)], [int(fn), int(tp)]]}")
    print(
        f"{name}: accuracy {accuracy_score(y_test, pred):.6f}, precision {precision_score(y_test, pred):.6f}, "
        f"recall {recall_score(y_test, pred):.6f}, F1 {f1_score(y_test, pred):.6f}"
    )


def main():
    x_train, x_test, y_train, y_test = split_dataset(load_dataset())
    train, test = x_train.tolist(), x_test.tolist()

    nb = build_pipeline("nb", "word").fit(train, y_train)
    svm = build_pipeline("svm", "word+char").fit(train, y_train)
    DOCS.mkdir(exist_ok=True)

    nb_path = write_json(DOCS / "model.json", export_nb(nb.named_steps["tfidf"], nb.named_steps["clf"]))
    svm_path = write_json(DOCS / "model-svm.json", export_svm(svm, n_docs=len(train)))
    fixtures_path = write_rows(DOCS / "fixtures.json", build_fixtures(test, y_test, nb, svm))
    analyzer_path = write_rows(DOCS / "fixtures-analyzer.json", analyzer_fixtures(svm, test[:40]))

    print(f"Naive Bayes vocabulary size: {len(nb.named_steps['tfidf'].vocabulary_)}")
    print(f"SVM features: {len(svm.named_steps['clf'].coef_[0])}")
    for path in (nb_path, svm_path, fixtures_path, analyzer_path):
        print(f"wrote {path.relative_to(ROOT)} ({size(path)})")
    print(f"test messages: {len(test)} (spam: {int(y_test.sum())}, ham: {int((y_test == 0).sum())})")
    report("Naive Bayes", y_test, nb.predict(test))
    report("Linear SVM", y_test, svm.predict(test))
    print(f"majority baseline: {accuracy_score(y_test, [0] * len(test)):.6f} (always predict ham)")


if __name__ == "__main__":
    main()
