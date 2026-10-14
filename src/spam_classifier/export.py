"""Export trained models to JSON for the in-browser demo, with the fixtures that test them.

Two models are exported:

* the notebook's Naive Bayes (word TF-IDF): ``docs/model.json``, unchanged since the first demo;
* the best model of the comparison, a linear SVM on word and character TF-IDF: ``docs/model-svm.json``.

The SVM file is kept small. The inverse document frequencies are not stored; the demo rebuilds them from
integer document frequencies with scikit-learn's smoothed formula ``ln((1 + n) / (1 + df)) + 1``. The
coefficients are rounded to 6 decimals, which moves the decision values by about 2e-6 on the test messages,
far less than the smallest margin (0.0009). The two vectorizers of the ``FeatureUnion`` keep their own
vocabularies and their own L2 normalisation, as in scikit-learn.
"""

import random

import numpy as np

# Inputs on which the JavaScript character n-gram analyzer is checked against scikit-learn. They cover the
# places where Python and JavaScript could differ: which characters count as white space, characters outside
# the Basic Multilingual Plane, case mapping, one-letter and empty messages.
AWKWARD_TEXTS = [
    "",
    "   ",
    "I",
    "Hi",
    "A",
    "a  b",
    "Free  entry\tin 2 a wkly comp",
    "line one\nline two\r\nline three",
    "WIN \xa31000 NOW!!! Call 08001234567",
    "caf\xe9 \xe0 la carte, na\xefve",
    "İstanbul ǅ \xdf ẞ STRASSE",
    "ΣΑΣ final sigma",
    "emoji \U0001f600 here \U0001f600\U0001f600",
    "nel\x85split",
    "nbsp\xa0split",
    "ideographic　space",
    "zero​width joiner",
    "﻿BOM at the start",
    "\x1ctest\x1dseparators",
    "tab\tseparated\tvalues",
    "a" * 60,
    "x" * 3 + " " + "y" * 2,
    "http://www.example.com/win?id=1&x=2",
    "&lt;#&gt; ok",
    "UPPER lower MiXeD",
    "你好 世界",
]


_ALPHABET = list(
    "abcXYZ019 !?.,'\"-_&#/@\u00a3$\xe9\xdf\u0130\u03a3\u4f60\U0001f600          \t\n\r\x85\xa0\u3000\u200b\ufeff\x1c"
)


def random_texts(n=300, seed=0):
    """Seeded random strings mixing letters, digits, punctuation and every kind of white space."""
    rng = random.Random(seed)
    return ["".join(rng.choices(_ALPHABET, k=rng.randint(1, 40))) for _ in range(n)]


def export_nb(vec, clf):
    """The Naive Bayes model as JSON-ready data (the format docs/predict.mjs reads)."""
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


def _document_frequencies(vec, n_docs):
    """Integer document frequencies that rebuild ``vec.idf_`` exactly with the smoothed formula."""
    if not vec.smooth_idf:
        raise ValueError("only smooth_idf=True vectorizers can be exported this way")
    df = np.rint((1 + n_docs) / np.exp(vec.idf_ - 1) - 1).astype(int)
    rebuilt = np.log((1 + n_docs) / (1 + df)) + 1
    if not np.allclose(rebuilt, vec.idf_, rtol=0, atol=1e-12):
        raise ValueError("idf cannot be rebuilt from integer document frequencies; is n_docs right?")
    return df.tolist()


def _params(vec, names):
    """Selected vectorizer parameters, with tuples as lists, as they read back from JSON."""
    p = vec.get_params()
    return {k: list(p[k]) if isinstance(p[k], tuple) else p[k] for k in names}


def _terms(vec):
    return sorted(vec.vocabulary_, key=vec.vocabulary_.get)


def export_svm(pipe, n_docs, decimals=6):
    """The word + character TF-IDF linear SVM as JSON-ready data (the format docs/svm.mjs reads).

    ``n_docs`` is the number of training messages the vectorizers were fitted on.
    """
    parts = dict(pipe.named_steps["tfidf"].transformer_list)
    clf = pipe.named_steps["clf"]
    word, char = parts["word"], parts["char"]
    return {
        "model": "svm",
        "classes": clf.classes_.tolist(),
        "intercept": float(clf.intercept_[0]),
        "word": {
            "params": _params(word, ("lowercase", "ngram_range", "sublinear_tf", "norm", "token_pattern")),
            "stop_words": sorted(word.get_stop_words() or []),
            "terms": _terms(word),
            "df": _document_frequencies(word, n_docs),
            "n_docs": int(n_docs),
        },
        "char": {
            "params": _params(char, ("lowercase", "analyzer", "ngram_range", "sublinear_tf", "norm")),
            "terms": _terms(char),
            "df": _document_frequencies(char, n_docs),
            "n_docs": int(n_docs),
        },
        # word features first, then character features: the order of the FeatureUnion
        "coef": np.round(clf.coef_[0], decimals).tolist(),
    }


def analyzer_fixtures(pipe, sample_texts):
    """Character n-grams from scikit-learn for awkward inputs and some real messages."""
    analyze = dict(pipe.named_steps["tfidf"].transformer_list)["char"].build_analyzer()
    texts = list(dict.fromkeys(AWKWARD_TEXTS + list(sample_texts) + random_texts()))
    return [{"text": t, "ngrams": analyze(t)} for t in texts]


def build_fixtures(texts, labels, nb_pipe, svm_pipe):
    """One row per test message: the label and both scikit-learn models' predictions."""
    texts = list(texts)
    nb_pred = nb_pipe.predict(texts)
    svm_pred = svm_pipe.predict(texts)
    svm_score = svm_pipe.decision_function(texts)
    return [
        {
            "text": text,
            "label": int(label),
            "sklearn": int(a),
            "svm": int(b),
            "svm_score": round(float(s), 9),
        }
        for text, label, a, b, s in zip(texts, labels, nb_pred, svm_pred, svm_score, strict=True)
    ]
