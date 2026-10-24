# Data preparation

Raw datasets are **not included** in this repository (licences/size; and I could not download them from the
environment where this project was built). Place the files exactly as below, then run `python main.py check`.

```
data/raw/accidents/   <- ONE accident/collision CSV (any file name, *.csv)
data/raw/images/      <- labelled images, one sub-folder per class:  images/<class_name>/*.jpg
data/raw/match_pair/  <- TWO related images of the same scene (different angle/lighting)
```

## A. Accident data (recommended: UK Department for Transport "Road Safety Data")
1. Open the DfT road-safety open-data page (see `LICENSE_AND_DATA_SOURCES.md`) and download one year of the
   **collision / accident** CSV (e.g. a single recent year, roughly 100k rows - small enough for Colab).
2. Put the CSV in `data/raw/accidents/`. Do **not** edit it.

The loader accepts both the older `accident_*` and newer `collision_*` column names. Required columns:
`accident_severity` (1 fatal, 2 serious, 3 slight). Used if present: `date`, `time`, `day_of_week`, `speed_limit`,
`number_of_vehicles`, `latitude`, `longitude`, `road_type`, `junction_detail`, `light_conditions`,
`weather_conditions`, `road_surface_conditions`, `urban_or_rural_area`, and similar coded fields.
Category codes are numeric; their meanings are in the DfT "road safety data guide" distributed with the data.

Another accident dataset can be used if it has a severity column - then adapt `config.py` (feature lists,
`CODE_COLUMNS`) and the column mapping in `src/data_loader.py`.

## B. Images
Choose a public motorcycle / road-scene image dataset whose licence allows study use (record it in
`LICENSE_AND_DATA_SOURCES.md`). Requirements:
* classification: at least 2 classes with >= 20 images each (more is better), in `data/raw/images/<class>/`;
* if the images are augmented copies of the same photo (e.g. Roboflow exports named `name_jpg.rf.<hash>.jpg`),
  keep the original file naming - the split keeps all copies of one source photo on the same side. Adjust
  `GROUP_SPLIT_REGEX` in `config.py` for other naming schemes;
* feature matching: two photos of the *same* scene/object from slightly different viewpoints in `data/raw/match_pair/`
  (you can photograph one yourself).

Kaggle datasets need a free account: `pip install kaggle`, create an API token in your Kaggle account settings,
save it as `~/.kaggle/kaggle.json` (never commit it), then `kaggle datasets download -d <owner>/<dataset> -p data/raw/... --unzip`.
