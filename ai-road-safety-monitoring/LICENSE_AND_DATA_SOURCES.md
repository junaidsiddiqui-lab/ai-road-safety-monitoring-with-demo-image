# Data sources, licences and limitations

> **Status:** these entries were prepared WITHOUT internet access, so no link, licence text, row count or image
> count could be verified or measured. Fields marked **TO FILL** must be completed by the student after
> downloading. Nothing below is an experimental result.

## A. Accident dataset (selected)
| Item | Value |
|---|---|
| Name | UK Department for Transport - Road Safety Data (STATS19 collision/accident records) |
| Publisher pages (confirm they still resolve) | https://www.gov.uk/government/statistical-data-sets/road-safety-open-data and https://www.gov.uk/government/collections/road-accidents-and-safety-statistics |
| Licence | Published by DfT as open data, normally under the **Open Government Licence v3.0** - **confirm on the download page** and record the exact wording here: TO FILL |
| Year/file downloaded | TO FILL |
| Rows x columns (raw) | TO FILL - printed by `python main.py clean`, stored in `outputs/reports/cleaning_summary.json` |
| Target | `accident_severity` (1 fatal, 2 serious, 3 slight) -> binary `severe` = fatal or serious |
| Known limitations | police-reported accidents only (slight injuries under-reported); coded categories; one country; no exposure data (traffic volume), so rates are not per-journey risk |

## B. Image dataset
| Item | Value |
|---|---|
| Name / source URL | **NOT YET SELECTED** - TO FILL. I could not browse dataset sites, so I did not name one I could not verify. |
| Licence | TO FILL |
| Image count and class labels | TO FILL - printed by `python main.py check` / `outputs/metrics/image_classification.json` |
| Selection checklist | clear usage licence; >= 2 labelled classes with >= 20 images each; source-photo information in file names (to avoid train/test leakage); a pair of related views of one scene for matching |
| Known limitations | TO FILL after inspecting the data |

## Relationship between A and B
The datasets are independent. No image is tied to an accident record, and nothing in this project infers one.


## Verified official source reference
The UK Department for Transport's official road safety open-data page states that its files cover police-reported personal-injury road collisions in Great Britain and lists the 2025 collision CSV (19.3 MB as shown on the page when checked on 25 October 2026):
https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

Direct CSV URL used by `download_accidents.py`:
https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2025.csv

The page identifies the data as coded records; consult the official data guide before interpreting code values. The data source is not motorcycle-only and does not contain linked road images.
