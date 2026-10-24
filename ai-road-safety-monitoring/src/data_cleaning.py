"""Reproducible cleaning pipeline for the accident dataset. The raw file is never modified.
Every decision is logged as: problem found -> method -> why."""
import numpy as np
import pandas as pd
import config


def _log(log, problem, method, why, **detail):
    log.append({"problem": problem, "method": method, "why": why, **detail})


def inspect(df):
    return {"rows": int(len(df)), "columns": int(df.shape[1]),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_share": {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0},
            "duplicate_rows": int(df.duplicated().sum())}


def clean_accidents(df_raw):
    """Return (clean_df, summary). Learned/statistical imputation is NOT done here (it happens inside the
    modelling pipeline, fitted on training data only) so no test information leaks."""
    log, df = [], df_raw.copy()
    summary = {"raw": inspect(df_raw)}

    # 1. Text clean-up and missing-value tokens
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
    df = df.replace(config.MISSING_TOKENS, np.nan)
    _log(log, "Whitespace and text tokens such as 'Data missing or out of range' hide real missing values",
         "Stripped text and replaced tokens with NaN", "Missing values must be recognised before they can be analysed")

    # 2. Sentinel -1 in coded columns
    replaced = 0
    for c in config.CODE_COLUMNS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            m = df[c] == -1
            replaced += int(m.sum())
            df.loc[m, c] = np.nan
    _log(log, "Coded columns use -1 for 'data missing or out of range'", "Replaced -1 with NaN in coded columns",
         "-1 is not a real category; treating it as a number would distort statistics", values_replaced=replaced)

    # 3. Duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    n_full = n0 - len(df)
    n_idx = 0
    if "accident_index" in df:
        before = len(df)
        df = df.drop_duplicates(subset="accident_index", keep="first")
        n_idx = before - len(df)
    _log(log, "Duplicate records inflate counts and can leak between train/test sets",
         "Dropped exact duplicate rows, then repeated accident_index values (keep first)",
         "accident_index uniquely identifies an accident record", full_duplicates=n_full, duplicate_index=n_idx)

    # 4. Data types
    for c in ["number_of_vehicles", "number_of_casualties", "latitude", "longitude", "speed_limit"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    date = pd.to_datetime(df["date"], dayfirst=True, errors="coerce") if "date" in df else pd.Series(pd.NaT, index=df.index)
    hour = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour if "time" in df else pd.Series(np.nan, index=df.index)
    _log(log, "Dates, times and counts are stored as text", "Parsed dates (day-first) and HH:MM times, coerced numbers",
         "Needed for time features; unparseable values become NaN instead of crashing",
         unparsable_dates=int(date.isna().sum()), unparsable_times=int(hour.isna().sum()))

    # 5. Impossible coordinates
    bad = 0
    for c, lim in [("latitude", 90), ("longitude", 180)]:
        if c in df:
            m = df[c].abs() > lim
            bad += int(m.sum())
            df.loc[m, c] = np.nan
    _log(log, "Coordinates outside the valid range", "Set to NaN", "Impossible values are errors, not outliers", coordinates_invalid=bad)

    # 6. Outliers: detect and document, never delete automatically
    outliers = {}
    for c in config.OUTLIER_COLUMNS:
        if c in df and df[c].notna().any():
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers[c] = {"lower": float(lo), "upper": float(hi), "count": int(((df[c] < lo) | (df[c] > hi)).sum()),
                           "max": float(df[c].max())}
    _log(log, "Extreme counts (many vehicles/casualties) found with the 1.5xIQR rule",
         "Kept the rows; reduced skew with a log1p transform inside the modelling pipeline",
         "Large multi-vehicle crashes are real events, and they are the rare serious ones we care about", outliers=outliers)

    # 7. Feature engineering
    weekday = date.dt.weekday
    if "day_of_week" in df:  # DfT coding: 1=Sunday ... 7=Saturday -> 0=Monday ... 6=Sunday
        weekday = weekday.fillna((df["day_of_week"] - 2) % 7)
    df["hour"] = hour
    df["month"] = date.dt.month
    df["year"] = date.dt.year
    df["day_of_week"] = weekday
    df["is_weekend"] = np.where(weekday.isna(), np.nan, (weekday >= 5).astype(float))
    df["is_rush_hour"] = np.where(hour.isna(), np.nan, hour.isin([7, 8, 9, 16, 17, 18]).astype(float))
    df["is_night"] = np.where(hour.isna(), np.nan, ((hour < 6) | (hour >= 21)).astype(float))
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    _log(log, "Raw date/time cannot be used directly by models",
         "Created hour, month, weekday, weekend/rush-hour/night flags and cyclic hour_sin/hour_cos",
         "Hour 23 and hour 0 are neighbours; the cyclic encoding preserves that")

    # 8. Categorical codes -> text categories (object dtype, NaN preserved)
    for c in config.CATEGORICAL_FEATURES:
        if c in df:
            df[c] = df[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan).astype(object)

    # 9. Target
    if "accident_severity" not in df:
        raise ValueError("Column 'accident_severity' (or 'collision_severity') is required to build the target.")
    n_before = len(df)
    df = df[df["accident_severity"].isin([1, 2, 3])].copy()
    df[config.TARGET] = df["accident_severity"].isin([1, 2]).astype(int)
    _log(log, "Severity has three levels (1 fatal, 2 serious, 3 slight) and fatal is very rare",
         "Binary target severe = fatal or serious", "Keeps enough positive cases to learn from; rows without severity are unusable for supervised learning",
         rows_dropped_no_target=n_before - len(df))

    # 10. Columns that are mostly empty
    share = df.isna().mean()
    drop = [c for c in share[share > config.MAX_MISSING_SHARE].index if c != config.TARGET]
    df = df.drop(columns=drop)
    _log(log, "Columns with more than 50% missing values carry too little information", "Dropped them (rows kept)",
         "Rows with some missing fields are retained; remaining gaps are imputed later inside the pipeline", dropped_columns=drop)

    summary.update({"steps": log, "clean": inspect(df),
                    "target_counts": {str(k): int(v) for k, v in df[config.TARGET].value_counts().items()},
                    "severe_share": round(float(df[config.TARGET].mean()), 4)})
    return df.reset_index(drop=True), summary
