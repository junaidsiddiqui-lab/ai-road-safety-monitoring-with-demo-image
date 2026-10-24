"""Exploratory data analysis on the cleaned accident data. Every figure is saved to outputs/figures/ and
every finding is computed from the data (nothing is hard-coded)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src import data_loader

T = config.TARGET
SEV = {0: "Slight", 1: "Fatal/Serious"}


def _save(fig, name):
    path = config.FIG_DIR / name
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path.name)


def _rate_table(df, col, min_n):
    g = df.groupby(col)[T].agg(rate="mean", n="count")
    return g[g["n"] >= min_n].sort_values("rate", ascending=False)


def run_eda(df):
    sns.set_theme(style="whitegrid")
    figs, findings = {}, []
    n = len(df); share = df[T].mean()
    min_n = max(30, int(0.002 * n))
    num = [c for c in config.NUMERIC_FEATURES + config.SKEWED_FEATURES if c in df]

    # Overview
    desc = df[num].describe().T
    desc.to_csv(config.METRICS_DIR / "descriptive_statistics.csv")

    # Target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df[T].map(SEV).value_counts()
    sns.barplot(x=vc.index, y=vc.values, ax=ax, hue=vc.index, legend=False)
    ax.set_xlabel("Accident severity class"); ax.set_ylabel("Number of accidents"); ax.set_title("Target distribution (class imbalance)")
    for i, v in enumerate(vc.values): ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    figs["target_distribution"] = _save(fig, "eda_target_distribution.png")
    findings.append(f"The cleaned dataset has {n:,} accidents; {share:.1%} are Fatal/Serious and {1 - share:.1%} Slight, so the classes are imbalanced and accuracy alone would be misleading.")

    # Univariate: histograms + density
    k = len(num); cols = 3; rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows)); axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, num):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, bins=30); ax.set_xlabel(c); ax.set_ylabel("Count"); ax.set_title(f"Distribution of {c}")
    for ax in axes[k:]: ax.axis("off")
    figs["histograms"] = _save(fig, "eda_histograms.png")

    # Boxplots by severity
    box = [c for c in ["number_of_vehicles", "speed_limit", "hour"] if c in df]
    fig, axes = plt.subplots(1, len(box), figsize=(4.2 * len(box), 4))
    for ax, c in zip(np.atleast_1d(axes), box):
        sns.boxplot(data=df.assign(sev=df[T].map(SEV)), x="sev", y=c, ax=ax, hue="sev", legend=False)
        ax.set_xlabel("Severity"); ax.set_ylabel(c); ax.set_title(f"{c} by severity")
    figs["boxplots"] = _save(fig, "eda_boxplots.png")
    if "number_of_vehicles" in df:
        v = df["number_of_vehicles"]
        findings.append(f"Number of vehicles is right-skewed: median {v.median():.0f}, 95th percentile {v.quantile(.95):.0f}, maximum {v.max():.0f}. These large crashes are real events, so they were kept and a log transform is used for modelling.")

    # Categorical features vs target
    cats = [c for c in ["light_conditions", "weather_conditions", "road_surface_conditions", "urban_or_rural_area", "road_type", "speed_limit"] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7)); axes = axes.ravel()
    for ax, c in zip(axes, cats):
        t = _rate_table(df, c, min_n).sort_index()
        ax.bar(t.index.astype(str), t["rate"], color="tab:red"); ax.axhline(share, ls="--", color="k", lw=1, label="overall rate")
        ax.set_xlabel(f"{c} (code)"); ax.set_ylabel("Share Fatal/Serious"); ax.set_title(f"Severe-accident share by {c}"); ax.legend()
    for ax in axes[len(cats):]: ax.axis("off")
    figs["feature_vs_target"] = _save(fig, "eda_feature_vs_target.png")

    # Time trends
    if "hour" in df:
        h = df.groupby("hour")[T].agg(rate="mean", n="count")
        fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(h.index, h["n"], color="lightsteelblue"); ax.set_xlabel("Hour of day"); ax.set_ylabel("Number of accidents")
        ax2 = ax.twinx(); ax2.plot(h.index, h["rate"], color="crimson", marker="o"); ax2.set_ylabel("Share Fatal/Serious"); ax2.grid(False)
        ax.set_title("Accidents (bars) and severe share (line) by hour")
        figs["hourly"] = _save(fig, "eda_hourly_trend.png")
        hh = h[h["n"] >= min_n]
        if len(hh):
            top, low = hh["rate"].idxmax(), hh["rate"].idxmin()
            findings.append(f"The share of Fatal/Serious accidents is highest at hour {int(top)} ({hh.loc[top, 'rate']:.1%}, n={int(hh.loc[top, 'n']):,}) and lowest at hour {int(low)} ({hh.loc[low, 'rate']:.1%}); the busiest hour for accidents is {int(h['n'].idxmax())}:00.")
    if "month" in df and df["month"].notna().any():
        m = df.groupby("month")[T].agg(rate="mean", n="count")
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(m.index, m["n"], marker="o"); ax.set_xlabel("Month"); ax.set_ylabel("Number of accidents"); ax.set_title("Accidents per month")
        figs["monthly"] = _save(fig, "eda_monthly_trend.png")

    # Group findings
    for col, label in [("speed_limit", "speed limit"), ("light_conditions", "light-condition code"), ("urban_or_rural_area", "urban/rural code")]:
        if col in df:
            t = _rate_table(df, col, min_n)
            if len(t) >= 2:
                findings.append(f"By {label}, the severe share ranges from {t['rate'].iloc[-1]:.1%} (value {t.index[-1]}) to {t['rate'].iloc[0]:.1%} (value {t.index[0]}) among groups with at least {min_n} accidents.")
    if "is_weekend" in df:
        w = df.groupby("is_weekend")[T].mean()
        if len(w) == 2:
            findings.append(f"Weekend accidents are {w.get(1.0, np.nan):.1%} severe versus {w.get(0.0, np.nan):.1%} on weekdays.")

    # Correlation heatmap (numeric + binary + target)
    cc = [c for c in num + config.BINARY_FEATURES + [T] if c in df]
    corr = df[cc].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (Pearson) of numeric features and the severe flag")
    figs["correlation"] = _save(fig, "eda_correlation_heatmap.png")
    ct = corr[T].drop(T).abs().sort_values(ascending=False)
    if len(ct):
        findings.append(f"The strongest linear correlation with the severe flag is {ct.index[0]} (r={corr.loc[ct.index[0], T]:+.2f}); all correlations are weak-to-moderate and, being observational, do not show causation.")

    # Scatter / multivariate
    if {"latitude", "longitude"} <= set(df.columns):
        s = df.dropna(subset=["latitude", "longitude"]).sample(min(20000, n), random_state=config.RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(6, 6))
        for k_, c_ in [(0, "tab:blue"), (1, "tab:red")]:
            q = s[s[T] == k_]; ax.scatter(q["longitude"], q["latitude"], s=3, alpha=.35 if k_ == 0 else .8, c=c_, label=SEV[k_])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Accident locations (sample)"); ax.legend(markerscale=4)
        figs["map_scatter"] = _save(fig, "eda_location_scatter.png")
    if {"hour", "speed_limit"} <= set(df.columns):
        d = df.dropna(subset=["hour", "speed_limit"]).copy(); d["time_band"] = pd.cut(d["hour"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
        p = d.pivot_table(index="time_band", columns="speed_limit", values=T, aggfunc="mean", observed=True)
        fig, ax = plt.subplots(figsize=(8, 4)); sns.heatmap(p, annot=True, fmt=".2f", cmap="Reds", ax=ax)
        ax.set_xlabel("Speed limit"); ax.set_ylabel("Time of day"); ax.set_title("Severe share by time band and speed limit (multivariate)")
        figs["multivariate"] = _save(fig, "eda_multivariate_heatmap.png")

    # Missingness / anomalies
    miss = df.isna().mean().sort_values(ascending=False); miss = miss[miss > 0].head(10)
    if len(miss):
        fig, ax = plt.subplots(figsize=(7, 4)); miss[::-1].plot.barh(ax=ax); ax.set_xlabel("Share missing"); ax.set_ylabel("Column"); ax.set_title("Columns with the most missing values (after cleaning)")
        figs["missing"] = _save(fig, "eda_missing_values.png")
        findings.append(f"After cleaning, '{miss.index[0]}' has the most remaining missing values ({miss.iloc[0]:.1%}); these are imputed inside the modelling pipeline (training data only).")

    data_loader.write_json({"findings": findings, "figures": figs, "min_group_size": min_n}, config.EDA_FINDINGS)
    return findings, figs
