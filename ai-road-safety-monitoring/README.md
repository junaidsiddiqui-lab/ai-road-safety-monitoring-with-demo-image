# AI-Based Road Safety Monitoring and Accident Risk Analysis System

**Student:** Junaid Naeem | **Roll no.:** SU92-BCSFM-F26-041 | **University:** Superior University

One integrated project combining data cleaning, EDA, image processing, ORB feature detection/matching, traditional
image classification and tabular machine learning behind one Streamlit interface.

## Honest status of this delivery
| Item | Status |
|---|---|
| All source code, tests, Streamlit app, notebook, report generator | Written |
| Code executed end-to-end | Yes - **only on synthetic random data** (inside a temporary copy) to prove the pipeline runs. Those outputs were deleted. |
| Real datasets downloaded / real results generated | **No** - the build environment had no internet access. `outputs/` is empty on purpose. |
| Streamlit app | Syntax-checked only; Streamlit was not installable in the build environment, so it has not been launched. |
| Notebook | Code cells were executed as a script on synthetic data; the shipped `.ipynb` has **no saved outputs** until you run it on real data. |
| Dataset licences / links / counts | Must be completed by you (`LICENSE_AND_DATA_SOURCES.md`). |
| GitHub repository | Not created (see instructions below). |

After you add real data and run the steps below, every figure, table and number in the notebook, app and
`outputs/reports/final_report.md` is generated from that run.

## Architecture
```
data/raw/accidents ──> data_loader ─> data_cleaning ─> accidents_clean.csv ─┬─> eda ───────────> figures
                                                                            └─> preprocessing ─> accident_modeling ─> metrics, models
data/raw/images ────> image_processing ─> feature_detection (Canny, ORB, BFMatcher) ─> feature_matches
                                      └─> image_classification (ORB -> BoVW -> SVM/k-NN) ─> metrics
all results ──> reporting.py ──> outputs/reports/final_report.md ;  app.py and the notebook READ these files
```
`config.py` holds every path/setting; `main.py` runs stages; notebook and app reuse the same `src/` functions.

## Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Prepare data -> see `data/README.md`, then
```bash
python main.py check          # confirms files are found
python main.py all            # clean, eda, images, match, classify, model, report
# or stage by stage: clean | eda | images | match | classify | model | report
python -m pytest -q           # unit tests (synthetic data only)
streamlit run app.py          # interface
jupyter notebook notebooks/road_safety_analysis.ipynb   # Kernel > Restart & Run All
```
Stages whose data are missing are skipped with a clear message, and the report marks them "not available".

## Method summary
* **Cleaning** (`src/data_cleaning.py`): token/`-1` -> NaN, duplicates, dtype fixes, invalid coordinates, IQR outlier *documentation* (rows kept), time features, binary target (fatal/serious vs slight). Raw file never modified. Each step is logged as problem -> method -> why.
* **Leakage control**: casualty counts, police attendance and identifiers are excluded; imputation, scaling, log-transform and one-hot encoding are inside sklearn `Pipeline`/`ColumnTransformer` fitted on the training split only; stratified fixed-seed split.
* **Models**: Logistic Regression, k-NN, Decision Tree, Random Forest. Primary metric: F1 of the Fatal/Serious class (accuracy is misleading under imbalance). Train-vs-test gap is reported for over/underfitting.
* **Images**: grayscale -> Gaussian blur -> CLAHE; Canny (median-based thresholds); ORB (256-bit descriptors); BFMatcher with Hamming distance using cross-check and Lowe ratio test; RANSAC inlier count as a correctness proxy.
* **Image classification**: ORB descriptors -> k-means vocabulary (training images only) -> normalised histograms -> SVM / k-NN; group-aware split so copies of one source photo never straddle train/test.

## Limitations
Observational data (no causation); educational predictions only; no helmet-violation or live-video detection; classical features are weak under viewpoint/lighting change; results depend on the datasets you choose.

## Publishing to GitHub
1. Create an empty repository named `ai-road-safety-monitoring` at github.com (no README).
2. In the project folder:
```bash
git init && git add . && git commit -m "Initial commit: AI road safety monitoring project"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-road-safety-monitoring.git
git push -u origin main
```
3. The public link is `https://github.com/<your-username>/ai-road-safety-monitoring` once you have pushed.
Raw datasets and credentials are git-ignored; check `git status` before committing and never commit `kaggle.json`.
