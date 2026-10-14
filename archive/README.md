# Archive: legacy site files

Everything in this folder is **unrelated to the SMS spam classifier**. These files were in the repository root before it was restructured as an NLP project. They were moved here with `git mv` so that nothing is lost and the file history is kept. None of them is used by the code, the tests, the notebook or the demo.

## What is here

| Files | What they are |
|---|---|
| 17 `.jpeg` images | AI-generated illustrations, one per portfolio-style project title: Bank Note Authentication, Customer Segmentation Analysis with Tableau, Employee Performance Analytics with Power BI, Financial Statement Analysis Dashboard with Tableau, Housing Price Prediction, Inventory Management Dashboard with Tableau, Loan Prediction, Mall Customer Segmentation, Marketing Campaign Effectiveness Dashboard with Power BI, Movie Recommender System, Sales Performance Dashboard with Power BI, Stock Market Analysis and Prediction, Supply Chain Optimization Dashboard with Power BI, Twitter Sentiment Analysis, `future.jpeg`, `gan.jpeg` and `pink.jpeg`. They are decorative artwork. They are not charts, screenshots or model output, and they do not show results of any code in this repository. |
| `stylesheet.css` | The W3.CSS 4.15 style sheet from w3schools.com (third-party). Nothing in this project uses it. |
| `spam-ham.pdf` | The IEEE conference paper Şahin, Aydos and Orhan (2018), "Spam/Ham E-Mail Classification using Machine Learning Methods based on Bag of Words Technique" (https://doi.org/10.1109/SIU.2018.8404347). It is copyrighted by IEEE and its authors and is **not** covered by this repository's license. |
| `jekyll-gh-pages.yml` | The GitHub Actions workflow that GitHub suggests for Jekyll sites. It is GitHub's sample Jekyll deployment workflow, and its runs on `main` failed (2026-10-13 and 2026-10-14). It is kept here, out of `.github/workflows/` so it no longer runs. The live demo is served directly from `docs/` by GitHub Pages and does not need it. |

## Use in the project

One image, `Twitter Sentiment Analysis.jpeg`, is used as the decorative banner in the top-level README, with alt text that says what it is.

## License

`stylesheet.css` (W3.CSS) and the paper are third-party and keep their own licenses and copyright; see `THIRD_PARTY_NOTICES.md`. The repository does not record where the images came from.
