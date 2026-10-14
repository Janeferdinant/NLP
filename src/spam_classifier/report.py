"""The full model comparison, error analysis and threshold study (``evaluate --compare``).

Everything is computed from the notebook's split (``test_size=0.3``, ``random_state=0``): 3,900 training and
1,672 test messages. Models are chosen on the training split only (5-fold stratified cross-validation); the
test split is used once per model to report results. Hyperparameters are scikit-learn defaults and are not
tuned, so no model has seen the test messages in any way.

Outputs, with ``<date>`` the run date:
  <out_dir>/<date>-model-comparison.md   environment, results tables, thresholds, top terms
  <out_dir>/<date>-error-analysis.md     missed spam and false alarms, masked and shortened
  <out_dir>/<date>-results.json          the same numbers in machine-readable form
  <figures_dir>/pr_curve.png, model_comparison.png
"""

import datetime
import json
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from .data import RANDOM_STATE, TEST_SIZE, load_dataset, split_dataset  # noqa: E402
from .evaluation import (  # noqa: E402
    bootstrap_accuracy_ci,
    compute_metrics,
    default_threshold,
    error_cases,
    mask_digits,
    operating_point,
    out_of_fold_scores,
    spam_scores,
    threshold_for_precision,
    threshold_sweep,
    top_terms,
)
from .models import DISPLAY_NAMES, build_pipeline  # noqa: E402

FEATURE_LABELS = {"word": "word TF-IDF", "char": "char 2-5 TF-IDF", "word+char": "word + char TF-IDF"}
PROB_THRESHOLDS = [0.9, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
MARGIN_THRESHOLDS = [0.5, 0.0, -0.25, -0.5, -0.75]
BASELINE_KEY = "nb/word"
# Okabe-Ito colour-blind-safe palette
COLOURS = {"nb/word": "#0072B2", "logreg/word": "#D55E00", "svm/word": "#009E73", "best": "#CC79A7"}


@dataclass(frozen=True)
class Experiment:
    algorithm: str
    features: str
    normalize: bool = False
    class_weight: str | None = None
    primary: bool = True

    @property
    def key(self):
        suffix = ("+norm" if self.normalize else "") + ("+balanced" if self.class_weight else "")
        return f"{self.algorithm}/{self.features}{suffix}"

    @property
    def label(self):
        extra = ", normalised text" if self.normalize else ""
        extra += ", balanced class weights" if self.class_weight else ""
        return f"{DISPLAY_NAMES[self.algorithm]}, {FEATURE_LABELS[self.features]}{extra}"

    def build(self):
        return build_pipeline(self.algorithm, self.features, normalize=self.normalize, class_weight=self.class_weight)


PRIMARY = [Experiment(a, f) for f in ("word", "char", "word+char") for a in ("nb", "logreg", "svm")]
VARIANTS = [
    Experiment("nb", "word", normalize=True, primary=False),
    Experiment("logreg", "word", normalize=True, primary=False),
    Experiment("svm", "word", normalize=True, primary=False),
    Experiment("logreg", "word", class_weight="balanced", primary=False),
    Experiment("svm", "word", class_weight="balanced", primary=False),
]


def _cv(pipe, x, y):
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(clone(pipe), x, y, cv=folds, scoring=scoring)
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std())) for m in scoring}


def _run_experiment(exp, x_train, y_train, x_test, y_test):
    pipe = exp.build()
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    pred = pipe.predict(x_test)
    scores = spam_scores(pipe, x_test)
    return {
        "key": exp.key,
        "label": exp.label,
        "algorithm": exp.algorithm,
        "features": exp.features,
        "primary": exp.primary,
        "n_features": int(len(pipe.named_steps["tfidf"].get_feature_names_out())),
        "fit_seconds": fit_seconds,
        "test": compute_metrics(y_test, pred),
        "accuracy_ci95": bootstrap_accuracy_ci(y_test, pred, seed=RANDOM_STATE),
        "average_precision": float(average_precision_score(y_test, scores)),
        "cv": _cv(pipe, x_train, y_train),
        "_pipe": pipe,
        "_pred": pred,
        "_scores": scores,
    }


def _short(text, limit=160):
    text = " ".join(mask_digits(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _code(text):
    return "`" + text.replace("`", "'") + "`"


_URL = re.compile(r"(?i)(https?://|www\.|\b\w+\.(com|co\.uk|net|org)\b)")
_LONG_NUMBER = re.compile(r"\d{7,}|\d{5}[ -]?\d{3,}")
_MONEY = re.compile(r"[\xa3$€]")


def _group_stats(texts):
    texts = list(texts)
    if not texts:
        return {"n": 0, "median_chars": None, "pct_url": None, "pct_phone": None, "pct_money": None}

    def pct(rx):
        return 100 * sum(bool(rx.search(t)) for t in texts) / len(texts)

    return {
        "n": len(texts),
        "median_chars": float(np.median([len(t) for t in texts])),
        "pct_url": pct(_URL),
        "pct_phone": pct(_LONG_NUMBER),
        "pct_money": pct(_MONEY),
    }


def _thresholds(pipe):
    return PROB_THRESHOLDS if hasattr(pipe, "predict_proba") else MARGIN_THRESHOLDS


def _threshold_study(res, x_train, y_train, y_test):
    pipe, scores = res["_pipe"], res["_scores"]
    study = {
        "default_threshold": default_threshold(pipe),
        "sweep": threshold_sweep(y_test, scores, _thresholds(pipe)),
    }
    try:
        t = threshold_for_precision(y_test, scores, 1.0)
        study["max_recall_without_false_positives_on_test"] = operating_point(y_test, scores, t)
    except ValueError:
        study["max_recall_without_false_positives_on_test"] = None
    oof = out_of_fold_scores(pipe, x_train, y_train, seed=RANDOM_STATE)
    study["chosen_on_training_folds"] = {}
    for target in (1.0, 0.99):
        try:
            t = threshold_for_precision(y_train, oof, target)
        except ValueError:
            study["chosen_on_training_folds"][str(target)] = None
            continue
        row = operating_point(y_test, scores, t)
        row["target_precision_on_training_folds"] = target
        study["chosen_on_training_folds"][str(target)] = row
    return study


def _error_study(res, x_test, y_test):
    texts = x_test.tolist()
    fn, fp = error_cases(texts, y_test.tolist(), res["_pred"].tolist(), res["_scores"])
    y = np.asarray(y_test)
    pred = res["_pred"]
    caught = [t for t, yt, p in zip(texts, y, pred, strict=True) if yt == 1 and p == 1]
    ham_ok = [t for t, yt, p in zip(texts, y, pred, strict=True) if yt == 0 and p == 0]
    return {
        "fn": fn,
        "fp": fp,
        "stats": {
            "missed spam (false negatives)": _group_stats(e["text"] for e in fn),
            "caught spam (true positives)": _group_stats(caught),
            "flagged ham (false positives)": _group_stats(e["text"] for e in fp),
            "correct ham (true negatives)": _group_stats(ham_ok),
        },
    }


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:.0f}%"


def _metric_cells(r):
    """Accuracy, its 95% interval, precision, recall, F1, false positives and false negatives as table cells."""
    m = r["test"]
    lo, hi = r["accuracy_ci95"]
    return [
        f"{100 * m['accuracy']:.2f}%",
        f"{100 * lo:.2f} to {100 * hi:.2f}%",
        f"{m['precision']:.4f}",
        f"{m['recall']:.4f}",
        f"{m['f1']:.4f}",
        m["fp"],
        m["fn"],
    ]


def _cv_cell(r):
    mean, std = r["cv"]["f1"]
    return f"{mean:.4f} +/- {std:.4f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _environment(run_date):
    return [
        f"| Run date | {run_date} |",
        f"| Python | {platform.python_version()} ({platform.system()}, CPU only) |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| numpy | {np.__version__} |",
        f"| pandas | {pd.__version__} |",
        f"| matplotlib | {matplotlib.__version__} |",
    ]


def _plot_pr(results, best_key, y_test, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    shown = ["nb/word", "logreg/word", "svm/word"]
    if best_key not in shown:
        shown.append(best_key)
    for key in shown:
        res = results[key]
        colour = COLOURS.get(key, COLOURS["best"])
        prec, rec, _ = precision_recall_curve(y_test, res["_scores"])
        ax1.plot(rec, prec, color=colour, lw=2, label=f"{res['label']} (AP {res['average_precision']:.3f})")
        point = operating_point(y_test, res["_scores"], default_threshold(res["_pipe"]))
        ax1.plot(point["recall"], point["precision"], "o", color=colour, mec="black", ms=8)
    ax1.set_xlabel("Recall (share of spam caught)")
    ax1.set_ylabel("Precision (share of flagged that is spam)")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0.6, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Precision-recall curves, 1,672 test messages\n(dot = default decision threshold)", fontsize=10)

    nb = results["nb/word"]
    thresholds = np.logspace(-6, 0, 61)
    rows = threshold_sweep(y_test, nb["_scores"], thresholds)
    ax2.plot(thresholds, [r["recall"] for r in rows], color="#0072B2", lw=2, label="Recall")
    ax2.plot(thresholds, [r["precision"] for r in rows], color="#D55E00", lw=2, ls="--", label="Precision")
    ax2.axvline(0.5, color="gray", lw=1, ls=":")
    ax2.text(0.47, 0.3, "default 0.5", rotation=90, ha="right", va="center", fontsize=8, color="gray")
    ax2.set_xscale("log")
    ax2.set_xlabel("Spam-probability threshold (flag if probability >= threshold)")
    ax2.set_ylabel("Value")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_title("Naive Bayes baseline: moving the threshold", fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_comparison(results, out_path):
    prim = sorted((r for r in results.values() if r["primary"]), key=lambda r: r["test"]["f1"])
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    labels = [r["label"].replace(", ", "\n", 1) for r in prim]
    colour = {"nb": "#0072B2", "logreg": "#D55E00", "svm": "#009E73"}
    bars = ax.barh(range(len(prim)), [r["test"]["f1"] for r in prim], color=[colour[r["algorithm"]] for r in prim])
    ax.set_yticks(range(len(prim)), labels, fontsize=8)
    for bar, r in zip(bars, prim, strict=True):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{r['test']['f1']:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 for spam on the 1,672 test messages")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _sweep_table(study, status):
    rows = []
    for r in study["sweep"]:
        rows.append([f"{r['threshold']:g}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", r["fp"], r["fn"], status])
    return _table(["Threshold", "Precision", "Recall", "False positives", "False negatives", "Status"], rows)


def _op_row(label, r, status):
    return [label, f"{r['threshold']:.6g}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", r["fp"], r["fn"], status]


def _terms_table(spam, ham, status):
    rows = [
        [i + 1, _code(repr(s[0])), f"{s[1]:.2f}", _code(repr(h[0])), f"{h[1]:.2f}", status]
        for i, (s, h) in enumerate(zip(spam, ham, strict=True))
    ]
    return _table(["#", "Spam term", "Weight", "Ham term", "Weight", "Status"], rows)


def _comparison_md(run_date, status, results, best_key, studies, terms, y_test, n_train):
    n_test, n_spam_test = len(y_test), int(y_test.sum())
    L = []
    L.append(f"# Model comparison, run {run_date}\n")
    L.append(
        f"Status: **{status}**. These are new numbers from a new run. They do not replace the earlier notebook "
        "result; the Naive Bayes + word TF-IDF row reproduces that result (96.41% accuracy) with the package code.\n"
    )
    L.append("## Environment\n")
    L += _table(["Item", "Value"], [r.strip("| ").split(" | ") for r in _environment(run_date)])
    L.append("")
    L.append("## Method\n")
    L.append(
        f"- Split: `train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE})`, no stratification: "
        f"{n_train:,} training and {n_test:,} test messages ({n_spam_test} spam, {n_test - n_spam_test:,} ham) "
        "(same split as the notebook).\n"
        "- Features: word TF-IDF (the notebook's settings: lowercase, English stop words, unigrams); character "
        "TF-IDF (`char_wb`, 2 to 5 characters, sub-linear tf, `min_df=2`); both together.\n"
        "- Models: `MultinomialNB()`, `LogisticRegression(max_iter=1000)` and `LinearSVC()`, all scikit-learn "
        "defaults otherwise. No hyperparameter was tuned.\n"
        "- Selection: the best of the nine primary models is picked by mean F1 (spam) in 5-fold stratified "
        "cross-validation on the **training** split only. The test split is used once per model to report results.\n"
        "- Accuracy interval: 95% percentile bootstrap, 1,000 resamples of the test messages (seed 0).\n"
        "- Spam is the positive class. Average precision (AP) is the area under the precision-recall curve.\n"
    )
    L.append(f"## Table 1: the nine primary models, test results ({status})\n")
    rows = [
        [r["label"], f"{r['n_features']:,}"]
        + _metric_cells(r)
        + [f"{r['average_precision']:.4f}", _cv_cell(r), f"{r['fit_seconds']:.2f}", status]
        for r in sorted((r for r in results.values() if r["primary"]), key=lambda r: -r["cv"]["f1"][0])
    ]
    L += _table(
        [
            "Model, features",
            "Features",
            "Accuracy",
            "Accuracy 95% CI",
            "Precision",
            "Recall",
            "F1",
            "FP",
            "FN",
            "AP",
            "CV F1 (5-fold, train)",
            "Fit s",
            "Status",
        ],
        rows,
    )
    L.append(
        "\nRows are ordered by cross-validated F1 on the training split. FP = false positives (ham flagged as spam); FN = false negatives (spam missed).\n"
    )
    best = results[best_key]
    L.append(f"**Best by the training-split selection rule: {best['label']}.**\n")
    L.append(f"## Table 2: variants, text normalisation and class weights ({status})\n")
    rows = [
        [r["label"]] + _metric_cells(r) + [f"{r['average_precision']:.4f}", _cv_cell(r), status]
        for r in results.values()
        if not r["primary"]
    ]
    L += _table(
        [
            "Model, features",
            "Accuracy",
            "Accuracy 95% CI",
            "Precision",
            "Recall",
            "F1",
            "FP",
            "FN",
            "AP",
            "CV F1 (5-fold, train)",
            "Status",
        ],
        rows,
    )
    L.append(
        "\n`normalised text` replaces URLs, prices and 7+ digit numbers by tokens before vectorizing. "
        "`balanced class weights` re-weight the classes inversely to their frequency.\n"
    )
    for key, heading in studies["order"]:
        study = studies[key]
        res = results[key]
        L.append(f"## Threshold study: {heading} ({res['label']})\n")
        scale = "probability" if hasattr(res["_pipe"], "predict_proba") else "SVM margin"
        L.append(
            f"The score is the {scale}; the model's own decision point is {study['default_threshold']}. Test messages: {n_test:,} ({n_spam_test} spam).\n"
        )
        L.append(f"### Threshold sweep on the test split ({status})\n")
        L += _sweep_table(study, status)
        L.append("")
        L.append(f"### Selected operating points ({status})\n")
        op_rows = []
        d = operating_point(y_test, res["_scores"], study["default_threshold"])
        op_rows.append(_op_row("Default threshold", d, status))
        m = study["max_recall_without_false_positives_on_test"]
        if m:
            op_rows.append(
                _op_row("Highest recall with 0 false positives (picked on the test split, optimistic)", m, status)
            )
        for target, row in study["chosen_on_training_folds"].items():
            if row:
                op_rows.append(_op_row(f"Threshold picked on training folds for precision >= {target}", row, status))
        L += _table(["Operating point", "Threshold", "Precision", "Recall", "FP", "FN", "Status"], op_rows)
        L.append("")
    L.append(f"## Strongest terms ({status})\n")
    for key, heading in terms["order"]:
        spam, ham = terms[key]
        L.append(f"### {heading}: {results[key]['label']}\n")
        weight = "log-probability ratio spam/ham" if results[key]["algorithm"] == "nb" else "model coefficient"
        L.append(
            f"Weight is the {weight}; larger magnitude means a stronger cue. Terms are quoted; in the combined model `word__` marks a word feature and `char__` a character n-gram (spaces inside n-grams are word boundaries).\n"
        )
        L += _terms_table(spam, ham, status)
        L.append("")
    L.append("## Files\n")
    L.append("- `docs/figures/pr_curve.png`, `docs/figures/model_comparison.png`")
    L.append(f"- `docs/runs/{run_date}-error-analysis.md`, `docs/runs/{run_date}-results.json`")
    L.append("- Reproduce with `python -m spam_classifier evaluate --compare`.\n")
    return "\n".join(L)


def _errors_md(run_date, status, results, best_key, errors, n_test):
    L = [f"# Error analysis, run {run_date}\n"]
    L.append(
        f"Status: **{status}**. Spam is the positive class; a *false negative* is spam that was let through, a *false positive* is ham that was flagged. "
        f"Test messages: {n_test:,}, from the notebook's split.\n"
    )
    L.append(
        "The messages below are quoted from `data/spam.csv` (UCI SMS Spam Collection, CC BY 4.0), which is already public in this repository. "
        "Digit runs of 7 or more digits are replaced by `[number]` and each message is cut at 160 characters. Scores are the model's spam score "
        "(probability, or SVM margin where the heading says so).\n"
    )
    if best_key != BASELINE_KEY:
        o = errors["overlap"]
        L.append(
            f"## Baseline against the best model ({status})\n\n"
            f"Of the {o['baseline_fn']} spam messages the baseline missed, the best model also misses {o['both']} and catches "
            f"{o['baseline_fn'] - o['both']}. The best model misses {o['best_only']} spam messages that the baseline caught.\n"
        )
    for key, heading in errors["order"]:
        e = errors[key]
        res = results[key]
        L.append(f"## {heading}: {res['label']}\n")
        L.append(f"Counts: {len(e['fn'])} false negatives, {len(e['fp'])} false positives. ({status})\n")
        L.append(f"### What the mistakes look like ({status})\n")
        rows = []
        for name, s in e["stats"].items():
            rows.append(
                [
                    name,
                    s["n"],
                    "n/a" if s["median_chars"] is None else f"{s['median_chars']:.0f}",
                    _fmt_pct(s["pct_url"]),
                    _fmt_pct(s["pct_phone"]),
                    _fmt_pct(s["pct_money"]),
                    status,
                ]
            )
        L += _table(
            [
                "Group",
                "Messages",
                "Median length (chars)",
                "With a link",
                "With a phone-style number",
                "With a currency sign",
                "Status",
            ],
            rows,
        )
        L.append("")
        L.append(f"### False positives: ham flagged as spam ({len(e['fp'])}, {status})\n")
        L += [f"- score {x['score']:.3f}: {_code(_short(x['text']))}" for x in e["fp"]] or ["None."]
        L.append("")
        L.append(f"### False negatives: spam let through ({len(e['fn'])}, {status})\n")
        L += [f"- score {x['score']:.3f}: {_code(_short(x['text']))}" for x in e["fn"]] or ["None."]
        L.append("")
    return "\n".join(L)


def _jsonable(res):
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    return out


def run_report(data_path=None, out_dir="docs/runs", figures_dir="docs/figures", run_date=None):
    run_date = run_date or datetime.date.today().isoformat()
    warnings.filterwarnings("ignore", category=UserWarning)
    status = f"Verified (run {run_date})"
    out_dir, figures_dir = Path(out_dir), Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    x_train_s, x_test_s, y_train_s, y_test_s = split_dataset(df)
    x_train, x_test = x_train_s.tolist(), x_test_s.tolist()
    y_train, y_test = y_train_s.to_numpy(), y_test_s.to_numpy()

    results = {}
    for exp in PRIMARY + VARIANTS:
        print(f"running {exp.key} ...", flush=True)
        results[exp.key] = _run_experiment(exp, x_train, y_train, x_test, y_test)

    best_key = max((k for k, r in results.items() if r["primary"]), key=lambda k: results[k]["cv"]["f1"][0])
    print(f"best by training-split cross-validation: {best_key}")

    focus = [(BASELINE_KEY, "Baseline from the notebook")]
    if best_key != BASELINE_KEY:
        focus.append((best_key, "Best model by the selection rule"))
    studies = {"order": focus}
    errors = {"order": focus}
    terms = {"order": focus}
    for key, _ in focus:
        print(f"threshold and error study for {key} ...", flush=True)
        studies[key] = _threshold_study(results[key], x_train, y_train, y_test)
        errors[key] = _error_study(results[key], x_test_s, y_test_s)
        terms[key] = top_terms(results[key]["_pipe"], 20)

    if best_key != BASELINE_KEY:
        miss = {k: set(np.nonzero((y_test == 1) & (results[k]["_pred"] == 0))[0]) for k, _ in focus}
        errors["overlap"] = {
            "baseline_fn": len(miss[BASELINE_KEY]),
            "both": len(miss[BASELINE_KEY] & miss[best_key]),
            "best_only": len(miss[best_key] - miss[BASELINE_KEY]),
        }

    _plot_pr(results, best_key, y_test, figures_dir / "pr_curve.png")
    _plot_comparison(results, figures_dir / "model_comparison.png")

    comparison = _comparison_md(run_date, status, results, best_key, studies, terms, y_test, len(x_train))
    (out_dir / f"{run_date}-model-comparison.md").write_text(comparison, encoding="utf-8", newline="\n")
    (out_dir / f"{run_date}-error-analysis.md").write_text(
        _errors_md(run_date, status, results, best_key, errors, len(x_test)), encoding="utf-8", newline="\n"
    )
    payload = {
        "status": status,
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "best_by_training_cv": best_key,
        "experiments": {k: _jsonable(r) for k, r in results.items()},
        "thresholds": {k: studies[k] for k, _ in focus},
        "error_counts": {
            k: {"fn": len(errors[k]["fn"]), "fp": len(errors[k]["fp"]), "stats": errors[k]["stats"]} for k, _ in focus
        },
    }
    (out_dir / f"{run_date}-results.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8", newline="\n"
    )
    print(f"wrote {out_dir / (run_date + '-model-comparison.md')}")
    print(f"wrote {out_dir / (run_date + '-error-analysis.md')}")
    print(f"wrote {out_dir / (run_date + '-results.json')}")
    print(f"wrote figures in {figures_dir}")
    return payload
