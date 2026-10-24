"""Shared configuration for the AI-Based Road Safety Monitoring System.
Every module, the notebook and the Streamlit app import paths and settings from here."""
import os
from pathlib import Path

STUDENT_NAME = "Junaid Naeem"
ROLL_NUMBER = "SU92-BCSFM-F26-041"
UNIVERSITY = "Superior University"
PROJECT_TITLE = "AI-Based Road Safety Monitoring and Accident Risk Analysis System"

# RSM_PROJECT_ROOT lets the tests run in a temporary folder without touching real data.
BASE_DIR = Path(os.environ.get("RSM_PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ACCIDENT_DIR = RAW_DIR / "accidents"      # put the accident CSV here
IMAGE_DIR = RAW_DIR / "images"            # one sub-folder per class label
MATCH_PAIR_DIR = RAW_DIR / "match_pair"   # two related images of the same scene
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
PROC_IMG_DIR = OUTPUT_DIR / "processed_images"
MATCH_DIR = OUTPUT_DIR / "feature_matches"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

CLEAN_FILE = PROCESSED_DIR / "accidents_clean.csv"
CLEANING_SUMMARY = REPORT_DIR / "cleaning_summary.json"
EDA_FINDINGS = REPORT_DIR / "eda_findings.json"
STATUS_FILE = REPORT_DIR / "run_status.json"
MODEL_FILE = MODEL_DIR / "best_accident_model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_MODEL_ROWS = 60000       # stratified subsample so KNN/RF stay fast on a laptop/Colab
TRAIN_EVAL_ROWS = 10000      # rows used to measure training-set performance

# ---- Accident data schema (UK DfT STATS19-style column names; see data/README.md) ----
MISSING_TOKENS = ["Data missing or out of range", "data missing or out of range", "", " ", "NA", "N/A", "null"]
CODE_COLUMNS = [  # coded categorical fields where -1 means "missing / out of range"
    "police_force", "accident_severity", "day_of_week", "local_authority_district", "first_road_class",
    "road_type", "speed_limit", "junction_detail", "junction_control", "second_road_class",
    "pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities", "light_conditions",
    "weather_conditions", "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards",
    "urban_or_rural_area", "did_police_officer_attend_scene_of_accident", "trunk_road_flag"]
OUTLIER_COLUMNS = ["number_of_vehicles", "number_of_casualties", "speed_limit"]
MAX_MISSING_SHARE = 0.5

TARGET = "severe"  # 1 = fatal or serious (DfT codes 1,2); 0 = slight (code 3)
NUMERIC_FEATURES = ["speed_limit", "latitude", "longitude", "hour_sin", "hour_cos", "month"]
SKEWED_FEATURES = ["number_of_vehicles"]
BINARY_FEATURES = ["is_weekend", "is_rush_hour", "is_night"]
CATEGORICAL_FEATURES = [
    "day_of_week", "road_type", "junction_detail", "junction_control", "first_road_class", "second_road_class",
    "pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities", "light_conditions",
    "weather_conditions", "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards",
    "urban_or_rural_area"]
# Known only AFTER the accident (target leakage) or identifiers / very high cardinality:
EXCLUDED_COLUMNS = {
    "number_of_casualties": "outcome known only after the accident (target leakage)",
    "did_police_officer_attend_scene_of_accident": "post-event information (leakage)",
    "accident_severity": "the raw target the label is built from",
    "accident_index": "identifier", "accident_reference": "identifier",
    "lsoa_of_accident_location": "very high-cardinality area code",
    "local_authority_district": "very high-cardinality area code",
    "local_authority_ons_district": "very high-cardinality area code",
    "local_authority_highway": "very high-cardinality area code",
    "police_force": "high-cardinality administrative code",
    "location_easting_osgr": "replaced by latitude/longitude", "location_northing_osgr": "replaced by latitude/longitude",
    "first_road_number": "high-cardinality identifier", "second_road_number": "high-cardinality identifier"}

# ---- Image settings ----
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ORB_FEATURES = 1000
CLASSIFY_ORB_FEATURES = 300
CLASSIFY_MAX_SIDE = 256
VOCAB_SIZE = 100
MIN_IMAGES_PER_CLASS = 20
# Images from the same source scene must stay in the same split. The group id is the file stem
# up to the first match of this regex (e.g. Roboflow exports: "img12_jpg.rf.<hash>" -> "img12").
GROUP_SPLIT_REGEX = r"\.rf\.|_jpg|_png|_aug\d*"


def ensure_dirs():
    for d in [ACCIDENT_DIR, IMAGE_DIR, MATCH_PAIR_DIR, PROCESSED_DIR, FIG_DIR, PROC_IMG_DIR, MATCH_DIR,
              METRICS_DIR, REPORT_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
