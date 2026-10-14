# Fresh-clone check of the README commands, 2026-10-14

Status: **Verified (run 2026-10-14)**. The commands in the README "Getting started" section were pasted, in order, into a fresh `git clone` of the branch (commit `34d2108`) with a new Python 3.11.9 virtual environment and `pip install -r requirements.txt` (Windows 11, CPU, Node.js 22.20). Nothing was changed between the clone and the run.

## Results

| Command | Result | Status |
|---|---|---|
| `pytest` | 60 passed in 17.65 s | Verified (run 2026-10-14) |
| `ruff check .` (ruff 0.17.0) | All checks passed | Verified (run 2026-10-14) |
| `python -m spam_classifier train --algorithm nb --features word` | accuracy 0.964115; precision 1.000000; recall 0.747899; F1 0.855769; confusion matrix `[[1434, 0], [60, 178]]` (the notebook's result, reproduced) | Verified (run 2026-10-14) |
| `python -m spam_classifier train` (best model: linear SVM, word + char) | accuracy 0.993421; precision 1.000000; recall 0.953782; F1 0.976344; confusion matrix `[[1434, 0], [11, 227]]` | Verified (run 2026-10-14) |
| `python -m spam_classifier evaluate` | the same numbers as the line above, from the saved model | Verified (run 2026-10-14) |
| `python -m spam_classifier predict "WINNER!! Claim your free prize now, call 08001234567" "Are we still meeting for lunch today?"` | `spam` (score 1.1521), `ham` (score -0.8987) | Verified (run 2026-10-14) |
| `python -m spam_classifier evaluate --compare --out-dir local-runs --figures-dir local-runs` | best by training-split cross-validation: `svm/word+char`; every test metric, cross-validation figure and threshold table identical to `docs/runs/2026-10-14-results.json` (only the fit-time column varies) | Verified (run 2026-10-14) |
| `jupyter nbconvert --to notebook --execute notebooks/spam_ham.ipynb --output-dir notebooks/executed` | executed copy prints `classifier accuracy 96.41%`; the committed notebook is untouched | Verified (run 2026-10-14) |
| `python scripts/export_model.py` | vocabulary size 6839; 1,672 test messages; confusion matrix `[[1434, 0], [60, 178]]`; `docs/model.json` and `docs/fixtures.json` rewritten with no content change (`git diff --exit-code` returns 0) | Verified (run 2026-10-14) |
| `node --test docs/parity.test.mjs` | `# pass 2`, `# fail 0`: 0 mismatches between the JavaScript classifier and scikit-learn on 1,672 messages | Verified (run 2026-10-14) |

## Tests written first

The test suite was committed before the package (commit `883edd8`). Running `pytest` at that commit, with `src/spam_classifier/` absent, stopped at collection with 8 errors (the package could not be imported). After the package was added (commit `0823163`) the same suite passed: 60 passed.

## Demo

The demo's files (`docs/index.html`, `docs/predict.mjs`, `docs/model.json`, `docs/fixtures.json`, `docs/parity.test.mjs`) are byte-identical to the merged version, because the demo still runs the Naive Bayes baseline. The best model of this repository (linear SVM on word and character TF-IDF) would need a different exported JSON and a JavaScript implementation of character n-grams, so the demo was not changed and was not re-checked in a browser.

## Environment

Python 3.11.9, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, joblib 1.6.0, matplotlib 3.11.2, pytest 9.1.1, ruff 0.17.0, nbconvert 7.17.2, ipykernel 7.4.0 (as pinned in `requirements.txt`).
