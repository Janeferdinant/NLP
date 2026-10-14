# Third-party notices

Original work in this repository is © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). The components below are third-party material and keep their own licenses.

## SMS Spam Collection (`data/spam.csv`)

| | |
|---|---|
| Creators | Tiago A. Almeida and José María Gómez Hidalgo |
| Source | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| DOI | https://doi.org/10.24432/C5CC84 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Changes | None to the messages. `data/spam.csv` in this repository is a copy with 5,572 rows (the UCI release lists 5,574 messages). The file is not valid UTF-8, so the notebook reads it as ISO-8859-1. The copy was inferred, from its columns and counts, to be a re-upload of the UCI data; the repository did not record where the copy came from. |

Suggested citation (from the UCI page): Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

The collection combines messages from several public sources (Grumbletext, the NUS SMS Corpus, a PhD thesis corpus and the SMS Spam Corpus v.0.1 Big); see the UCI page for details. The messages are public research data and contain phone numbers and shortcodes of the kind that appear in spam.

`docs/model.json` and `docs/fixtures.json` are derived from this dataset (a trained model's parameters, and held-out messages with predictions), so they are shared under the same attribution requirement (CC BY 4.0).

## Introductory paper for the dataset (cited, not included)

Almeida, T. A., Gómez Hidalgo, J. M. and Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering. https://doi.org/10.1145/2034691.2034742

## IEEE conference paper (cited, not redistributed)

Şahin, E., Aydos, M. and Orhan, F. (2018). Spam/Ham E-Mail Classification using Machine Learning Methods based on Bag of Words Technique. 2018 26th Signal Processing and Communications Applications Conference (SIU), pp. 1-4. https://doi.org/10.1109/SIU.2018.8404347

The PDF `archive/spam-ham.pdf` in this repository is that paper. It is copyrighted by IEEE and by its authors and is not covered by this repository's license. It is kept in `archive/` for reference only and is a candidate for removal by the owner; cite it by DOI.

## Libraries

The notebook and scripts use pandas, NumPy, scikit-learn, SciPy, joblib, Matplotlib and WordCloud. They are installed from PyPI under their own licenses (BSD-3-Clause or similar permissive licenses) and are not copied into this repository.

## `archive/stylesheet.css`

W3.CSS 4.15 (https://www.w3schools.com/w3css/), a third-party stylesheet that nothing in this repository uses. It is kept in `archive/` and keeps its own license.
