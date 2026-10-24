# Run Me First — AI-Based Road Safety Monitoring

## What is included
This repository contains the shared project code, one main notebook, a Streamlit app, tests, and the data-processing/modeling modules.

## Important status
The uploaded Claude ZIP did **not** contain the actual accident CSV or road/motorcycle images. Therefore, it cannot yet produce real experimental results. The existing automated tests passed in the current environment (6 passed), but those tests validate helper functions—not a full run on the real datasets.

## Step 1 — Install dependencies
From the extracted `ai-road-safety-monitoring` folder:
```bash
python -m pip install -r requirements.txt
```

## Step 2 — Download the real accident dataset
Run:
```bash
python download_accidents.py
```
This downloads the official UK Department for Transport collision dataset for 2025 into `data/raw/accidents/`. The source page documents personal-injury collisions reported to police in Great Britain: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

If the automatic download fails, use that official page to download **Road Safety Data - Collisions - 2025** and put the CSV in `data/raw/accidents/`.

## Step 3 — Add images
The image-processing, ORB feature-matching, and image-classification sections require actual images:
- `data/raw/match_pair/`: two related photos of the same scene/object for matching.
- `data/raw/images/<class_name>/`: labelled images arranged in one folder per class for image classification.

Choose a public image dataset whose license permits your use and record the dataset name, source, license, and image counts in `LICENSE_AND_DATA_SOURCES.md`. Do not claim the images are from the accident dataset unless they genuinely are linked.

## Step 4 — Run the project
```bash
python main.py check
python main.py clean
python main.py eda
python main.py model
python main.py report
```
Run image stages after adding images:
```bash
python main.py images
python main.py match
python main.py classify
```
The `all` command may report missing image inputs until those are supplied.

## Step 5 — Notebook and app
Open `notebooks/road_safety_analysis.ipynb` in Jupyter or Colab. Execute the cells from top to bottom after preparing the data.
Start the Streamlit app after results have been generated:
```bash
streamlit run app.py
```

## Before submission
Do not submit this as a completed real-data experiment until the dataset is added, the notebook is executed, and actual plots/metrics are generated. Confirm the notebook has saved outputs and the source/licence notes are completed.
