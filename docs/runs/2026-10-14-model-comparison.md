# Model comparison, run 2026-10-14

Status: **Verified (run 2026-10-14)**. These are new numbers from a new run. They do not replace the earlier notebook result; the Naive Bayes + word TF-IDF row reproduces that result (96.41% accuracy) with the package code.

## Environment

| Item | Value |
|---|---|
| Run date | 2026-10-14 |
| Python | 3.11.9 (Windows, CPU only) |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |

## Method

- Split: `train_test_split(test_size=0.3, random_state=0)`, no stratification: 3,900 training and 1,672 test messages (238 spam, 1,434 ham) (same split as the notebook).
- Features: word TF-IDF (the notebook's settings: lowercase, English stop words, unigrams); character TF-IDF (`char_wb`, 2 to 5 characters, sub-linear tf, `min_df=2`); both together.
- Models: `MultinomialNB()`, `LogisticRegression(max_iter=1000)` and `LinearSVC()`, all scikit-learn defaults otherwise. No hyperparameter was tuned.
- Selection: the best of the nine primary models is picked by mean F1 (spam) in 5-fold stratified cross-validation on the **training** split only. The test split is used once per model to report results.
- Accuracy interval: 95% percentile bootstrap, 1,000 resamples of the test messages (seed 0).
- Spam is the positive class. Average precision (AP) is the area under the precision-recall curve.

## Table 1: the nine primary models, test results (Verified (run 2026-10-14))

| Model, features | Features | Accuracy | Accuracy 95% CI | Precision | Recall | F1 | FP | FN | AP | CV F1 (5-fold, train) | Fit s | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear SVM, word + char TF-IDF | 40,299 | 99.34% | 98.92 to 99.70% | 1.0000 | 0.9538 | 0.9763 | 0 | 11 | 0.9971 | 0.9570 +/- 0.0131 | 0.85 | Verified (run 2026-10-14) |
| Linear SVM, char 2-5 TF-IDF | 33,460 | 99.10% | 98.62 to 99.52% | 1.0000 | 0.9370 | 0.9675 | 0 | 15 | 0.9978 | 0.9569 +/- 0.0111 | 0.76 | Verified (run 2026-10-14) |
| Linear SVM, word TF-IDF | 6,839 | 98.62% | 98.02 to 99.16% | 0.9864 | 0.9160 | 0.9499 | 3 | 20 | 0.9887 | 0.9198 +/- 0.0198 | 0.08 | Verified (run 2026-10-14) |
| Logistic regression, word + char TF-IDF | 40,299 | 97.85% | 97.13 to 98.50% | 1.0000 | 0.8487 | 0.9182 | 0 | 36 | 0.9945 | 0.9122 +/- 0.0154 | 0.97 | Verified (run 2026-10-14) |
| Logistic regression, char 2-5 TF-IDF | 33,460 | 97.01% | 96.17 to 97.79% | 1.0000 | 0.7899 | 0.8826 | 0 | 50 | 0.9960 | 0.8884 +/- 0.0165 | 0.84 | Verified (run 2026-10-14) |
| Multinomial Naive Bayes, word + char TF-IDF | 40,299 | 96.89% | 95.99 to 97.73% | 1.0000 | 0.7815 | 0.8774 | 0 | 52 | 0.9715 | 0.8752 +/- 0.0181 | 0.80 | Verified (run 2026-10-14) |
| Multinomial Naive Bayes, word TF-IDF | 6,839 | 96.41% | 95.51 to 97.31% | 1.0000 | 0.7479 | 0.8558 | 0 | 60 | 0.9734 | 0.8492 +/- 0.0217 | 0.06 | Verified (run 2026-10-14) |
| Logistic regression, word TF-IDF | 6,839 | 95.28% | 94.20 to 96.35% | 0.9818 | 0.6807 | 0.8040 | 3 | 76 | 0.9772 | 0.7798 +/- 0.0296 | 0.07 | Verified (run 2026-10-14) |
| Multinomial Naive Bayes, char 2-5 TF-IDF | 33,460 | 94.92% | 93.78 to 95.93% | 1.0000 | 0.6429 | 0.7826 | 0 | 85 | 0.9566 | 0.7685 +/- 0.0354 | 0.74 | Verified (run 2026-10-14) |

Rows are ordered by cross-validated F1 on the training split. FP = false positives (ham flagged as spam); FN = false negatives (spam missed).

**Best by the training-split selection rule: Linear SVM, word + char TF-IDF.**

## Table 2: variants, text normalisation and class weights (Verified (run 2026-10-14))

| Model, features | Accuracy | Accuracy 95% CI | Precision | Recall | F1 | FP | FN | AP | CV F1 (5-fold, train) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Multinomial Naive Bayes, word TF-IDF, normalised text | 97.01% | 96.17 to 97.79% | 1.0000 | 0.7899 | 0.8826 | 0 | 50 | 0.9834 | 0.8871 +/- 0.0187 | Verified (run 2026-10-14) |
| Logistic regression, word TF-IDF, normalised text | 96.29% | 95.33 to 97.19% | 0.9889 | 0.7479 | 0.8517 | 2 | 60 | 0.9864 | 0.8337 +/- 0.0145 | Verified (run 2026-10-14) |
| Linear SVM, word TF-IDF, normalised text | 98.80% | 98.27 to 99.28% | 0.9823 | 0.9328 | 0.9569 | 4 | 16 | 0.9932 | 0.9408 +/- 0.0113 | Verified (run 2026-10-14) |
| Logistic regression, word TF-IDF, balanced class weights | 98.39% | 97.73 to 98.98% | 0.9451 | 0.9412 | 0.9432 | 13 | 14 | 0.9790 | 0.9159 +/- 0.0131 | Verified (run 2026-10-14) |
| Linear SVM, word TF-IDF, balanced class weights | 98.74% | 98.15 to 99.22% | 0.9697 | 0.9412 | 0.9552 | 7 | 14 | 0.9871 | 0.9297 +/- 0.0133 | Verified (run 2026-10-14) |

`normalised text` replaces URLs, prices and 7+ digit numbers by tokens before vectorizing. `balanced class weights` re-weight the classes inversely to their frequency.

## Threshold study: Baseline from the notebook (Multinomial Naive Bayes, word TF-IDF)

The score is the probability; the model's own decision point is 0.5. Test messages: 1,672 (238 spam).

### Threshold sweep on the test split (Verified (run 2026-10-14))

| Threshold | Precision | Recall | False positives | False negatives | Status |
|---|---|---|---|---|---|
| 0.9 | 1.0000 | 0.3403 | 0 | 157 | Verified (run 2026-10-14) |
| 0.5 | 1.0000 | 0.7479 | 0 | 60 | Verified (run 2026-10-14) |
| 0.2 | 0.9442 | 0.9244 | 13 | 18 | Verified (run 2026-10-14) |
| 0.1 | 0.6845 | 0.9664 | 106 | 8 | Verified (run 2026-10-14) |
| 0.05 | 0.4701 | 0.9916 | 266 | 2 | Verified (run 2026-10-14) |
| 0.01 | 0.2109 | 0.9958 | 887 | 1 | Verified (run 2026-10-14) |
| 0.001 | 0.1466 | 1.0000 | 1385 | 0 | Verified (run 2026-10-14) |

### Selected operating points (Verified (run 2026-10-14))

| Operating point | Threshold | Precision | Recall | FP | FN | Status |
|---|---|---|---|---|---|---|
| Default threshold | 0.5 | 1.0000 | 0.7479 | 0 | 60 | Verified (run 2026-10-14) |
| Highest recall with 0 false positives (picked on the test split, optimistic) | 0.471417 | 1.0000 | 0.7773 | 0 | 53 | Verified (run 2026-10-14) |
| Threshold picked on training folds for precision >= 1.0 | 0.458156 | 0.9946 | 0.7773 | 1 | 53 | Verified (run 2026-10-14) |
| Threshold picked on training folds for precision >= 0.99 | 0.379464 | 0.9897 | 0.8067 | 2 | 46 | Verified (run 2026-10-14) |

## Threshold study: Best model by the selection rule (Linear SVM, word + char TF-IDF)

The score is the SVM margin; the model's own decision point is 0.0. Test messages: 1,672 (238 spam).

### Threshold sweep on the test split (Verified (run 2026-10-14))

| Threshold | Precision | Recall | False positives | False negatives | Status |
|---|---|---|---|---|---|
| 0.5 | 1.0000 | 0.8151 | 0 | 44 | Verified (run 2026-10-14) |
| 0 | 1.0000 | 0.9538 | 0 | 11 | Verified (run 2026-10-14) |
| -0.25 | 0.9788 | 0.9706 | 5 | 7 | Verified (run 2026-10-14) |
| -0.5 | 0.9402 | 0.9916 | 15 | 2 | Verified (run 2026-10-14) |
| -0.75 | 0.6910 | 0.9958 | 106 | 1 | Verified (run 2026-10-14) |

### Selected operating points (Verified (run 2026-10-14))

| Operating point | Threshold | Precision | Recall | FP | FN | Status |
|---|---|---|---|---|---|---|
| Default threshold | 0 | 1.0000 | 0.9538 | 0 | 11 | Verified (run 2026-10-14) |
| Highest recall with 0 false positives (picked on the test split, optimistic) | -0.0260117 | 1.0000 | 0.9622 | 0 | 9 | Verified (run 2026-10-14) |
| Threshold picked on training folds for precision >= 1.0 | 0.106164 | 1.0000 | 0.9328 | 0 | 16 | Verified (run 2026-10-14) |
| Threshold picked on training folds for precision >= 0.99 | -0.0422122 | 1.0000 | 0.9622 | 0 | 9 | Verified (run 2026-10-14) |

## Strongest terms (Verified (run 2026-10-14))

### Baseline from the notebook: Multinomial Naive Bayes, word TF-IDF

Weight is the log-probability ratio spam/ham; larger magnitude means a stronger cue. Terms are quoted; in the combined model `word__` marks a word feature and `char__` a character n-gram (spaces inside n-grams are word boundaries).

| # | Spam term | Weight | Ham term | Weight | Status |
|---|---|---|---|---|---|
| 1 | `'claim'` | 3.27 | `'ok'` | -3.55 | Verified (run 2026-10-14) |
| 2 | `'prize'` | 3.16 | `'lt'` | -3.34 | Verified (run 2026-10-14) |
| 3 | `'www'` | 2.98 | `'gt'` | -3.34 | Verified (run 2026-10-14) |
| 4 | `'150p'` | 2.83 | `'ll'` | -3.33 | Verified (run 2026-10-14) |
| 5 | `'uk'` | 2.78 | `'sorry'` | -3.24 | Verified (run 2026-10-14) |
| 6 | `'tone'` | 2.71 | `'later'` | -3.20 | Verified (run 2026-10-14) |
| 7 | `'guaranteed'` | 2.66 | `'lor'` | -3.13 | Verified (run 2026-10-14) |
| 8 | `'18'` | 2.65 | `'da'` | -3.02 | Verified (run 2026-10-14) |
| 9 | `'500'` | 2.52 | `'come'` | -3.02 | Verified (run 2026-10-14) |
| 10 | `'awarded'` | 2.50 | `'home'` | -2.88 | Verified (run 2026-10-14) |
| 11 | `'1000'` | 2.46 | `'ì_'` | -2.84 | Verified (run 2026-10-14) |
| 12 | `'16'` | 2.45 | `'wat'` | -2.79 | Verified (run 2026-10-14) |
| 13 | `'cs'` | 2.44 | `'going'` | -2.68 | Verified (run 2026-10-14) |
| 14 | `'service'` | 2.32 | `'doing'` | -2.58 | Verified (run 2026-10-14) |
| 15 | `'video'` | 2.30 | `'think'` | -2.57 | Verified (run 2026-10-14) |
| 16 | `'150ppm'` | 2.29 | `'lol'` | -2.55 | Verified (run 2026-10-14) |
| 17 | `'entry'` | 2.24 | `'oh'` | -2.53 | Verified (run 2026-10-14) |
| 18 | `'100'` | 2.21 | `'got'` | -2.49 | Verified (run 2026-10-14) |
| 19 | `'nokia'` | 2.19 | `'did'` | -2.46 | Verified (run 2026-10-14) |
| 20 | `'valid'` | 2.19 | `'yeah'` | -2.45 | Verified (run 2026-10-14) |

### Best model by the selection rule: Linear SVM, word + char TF-IDF

Weight is the model coefficient; larger magnitude means a stronger cue. Terms are quoted; in the combined model `word__` marks a word feature and `char__` a character n-gram (spaces inside n-grams are word boundaries).

| # | Spam term | Weight | Ham term | Weight | Status |
|---|---|---|---|---|---|
| 1 | `'word__uk'` | 1.22 | `'char__i '` | -0.45 | Verified (run 2026-10-14) |
| 2 | `'char__ 0'` | 1.14 | `'char__ i '` | -0.44 | Verified (run 2026-10-14) |
| 3 | `'char__00'` | 0.95 | `'char__ i'` | -0.43 | Verified (run 2026-10-14) |
| 4 | `'char__08'` | 0.86 | `'char__..'` | -0.43 | Verified (run 2026-10-14) |
| 5 | `'word__www'` | 0.85 | `'word__pa'` | -0.42 | Verified (run 2026-10-14) |
| 6 | `'word__txt'` | 0.77 | `'char__ m'` | -0.38 | Verified (run 2026-10-14) |
| 7 | `'char__50'` | 0.74 | `'char__ me '` | -0.38 | Verified (run 2026-10-14) |
| 8 | `'word__won'` | 0.70 | `'word__lt'` | -0.37 | Verified (run 2026-10-14) |
| 9 | `'word__claim'` | 0.69 | `'char__ me'` | -0.33 | Verified (run 2026-10-14) |
| 10 | `'char__0 '` | 0.68 | `'word__gt'` | -0.33 | Verified (run 2026-10-14) |
| 11 | `'char__ 1'` | 0.68 | `'word__fullonsms'` | -0.32 | Verified (run 2026-10-14) |
| 12 | `'char__80'` | 0.66 | `'word__yup'` | -0.31 | Verified (run 2026-10-14) |
| 13 | `'word__sexy'` | 0.65 | `'word__said'` | -0.30 | Verified (run 2026-10-14) |
| 14 | `'word__ac'` | 0.64 | `'char__:)'` | -0.30 | Verified (run 2026-10-14) |
| 15 | `'char__ 08'` | 0.64 | `'word__hey'` | -0.30 | Verified (run 2026-10-14) |
| 16 | `'char__ 8'` | 0.63 | `'word__plans'` | -0.30 | Verified (run 2026-10-14) |
| 17 | `'word__mobile'` | 0.62 | `'word__road'` | -0.30 | Verified (run 2026-10-14) |
| 18 | `'char__87'` | 0.61 | `'word__work'` | -0.30 | Verified (run 2026-10-14) |
| 19 | `'char__å£'` | 0.61 | `'char__me '` | -0.30 | Verified (run 2026-10-14) |
| 20 | `'word__146tf150p'` | 0.59 | `'word__mail'` | -0.29 | Verified (run 2026-10-14) |

## Files

- `docs/figures/pr_curve.png`, `docs/figures/model_comparison.png`
- `docs/runs/2026-10-14-error-analysis.md`, `docs/runs/2026-10-14-results.json`
- Reproduce with `python -m spam_classifier evaluate --compare`.
