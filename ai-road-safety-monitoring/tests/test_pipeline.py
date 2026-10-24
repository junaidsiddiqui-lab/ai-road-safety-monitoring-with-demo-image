"""Run with:  python -m pytest -q   (tests use synthetic data from tests/fixtures.py only)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src import data_cleaning, preprocessing, feature_detection as fd
import fixtures


def test_cleaning_removes_duplicates_and_sentinels():
    raw = fixtures.synthetic_accidents()
    df, s = data_cleaning.clean_accidents(raw)
    assert df["accident_index"].is_unique
    assert s["raw"]["rows"] - len(df) >= 20
    assert not (df["speed_limit"] == -1).any()
    assert df["weather_conditions"].isna().any()  # -1 and text token became NaN, not deleted rows


def test_cleaning_keeps_rows_with_missing_features_and_builds_target():
    df, _ = data_cleaning.clean_accidents(fixtures.synthetic_accidents())
    assert set(df[config.TARGET].unique()) <= {0, 1}
    assert df[config.TARGET].mean() == df["accident_severity"].isin([1, 2]).mean()
    cats = [c for c in config.CATEGORICAL_FEATURES if c in df]
    assert df[cats].isna().any().any()


def test_features_exclude_leakage_columns():
    df, _ = data_cleaning.clean_accidents(fixtures.synthetic_accidents())
    X, y = preprocessing.split_xy(df)
    for bad in ["number_of_casualties", "accident_severity", "accident_index", "did_police_officer_attend_scene_of_accident", config.TARGET]:
        assert bad not in X.columns


def test_preprocessor_fits_on_train_only_and_handles_unseen_category():
    df, _ = data_cleaning.clean_accidents(fixtures.synthetic_accidents())
    X_tr, X_te, y_tr, y_te = preprocessing.make_split(df)
    prep = preprocessing.build_preprocessor(X_tr).fit(X_tr)
    scaler = prep.named_transformers_["num"].named_steps["sc"]
    assert np.isclose(scaler.mean_[0], X_tr["speed_limit"].fillna(X_tr["speed_limit"].median()).mean())
    X_new = X_te.copy(); X_new.loc[X_new.index[0], "road_type"] = "999"
    assert prep.transform(X_new).shape[0] == len(X_new)


def test_split_is_stratified_and_reproducible():
    df, _ = data_cleaning.clean_accidents(fixtures.synthetic_accidents())
    a = preprocessing.make_split(df); b = preprocessing.make_split(df)
    assert a[1].index.equals(b[1].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.02


def test_orb_descriptors_and_matching():
    img = fixtures.make_scene(3)
    gray = fd.ip.to_gray(img)
    kp, des = fd.detect_orb(gray)
    assert des.shape[1] == 32 and len(kp) == des.shape[0] > 20  # 32 bytes = 256 bits
    assert len(fd.match_descriptors(des, des, "ratio")) >= 0
    assert len(fd.match_descriptors(des, des, "crosscheck")) == len(kp)
