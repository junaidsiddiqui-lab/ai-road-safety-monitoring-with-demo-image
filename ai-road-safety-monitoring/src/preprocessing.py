"""Leakage-safe preprocessing: all learned transformations live inside sklearn Pipelines, so they are
fitted on the training split only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import config


def feature_groups(df):
    """Feature types that actually exist in the dataframe."""
    pick = lambda cols: [c for c in cols if c in df.columns]
    return {"numeric": pick(config.NUMERIC_FEATURES), "skewed_numeric": pick(config.SKEWED_FEATURES),
            "binary": pick(config.BINARY_FEATURES), "categorical": pick(config.CATEGORICAL_FEATURES),
            "ordinal": pick(["speed_limit"]), "datetime_source": pick(["date", "time"]),
            "excluded": {k: v for k, v in config.EXCLUDED_COLUMNS.items() if k in df.columns}}


def split_xy(df):
    g = feature_groups(df)
    cols = g["numeric"] + g["skewed_numeric"] + g["binary"] + g["categorical"]
    return df[cols].copy(), df[config.TARGET].astype(int)


def build_preprocessor(X):
    g = feature_groups(X)
    steps = []
    if g["numeric"]:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), g["numeric"]))
    if g["skewed_numeric"]:
        steps.append(("skew", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                        ("sc", StandardScaler())]), g["skewed_numeric"]))
    if g["binary"]:
        steps.append(("bin", SimpleImputer(strategy="most_frequent"), g["binary"]))
    if g["categorical"]:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False))]),
                      g["categorical"]))
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def make_split(df, max_rows=None):
    """Stratified, reproducible subsample (if needed) followed by a stratified train/test split."""
    max_rows = max_rows or config.MAX_MODEL_ROWS
    if len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df[config.TARGET], random_state=config.RANDOM_STATE)
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
