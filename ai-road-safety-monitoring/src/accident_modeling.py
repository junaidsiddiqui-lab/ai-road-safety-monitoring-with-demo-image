"""Accident-severity classification: compares four models with leakage-safe pipelines."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import config
from src import data_loader, evaluation, preprocessing

RS = config.RANDOM_STATE
LABELS = ["Slight", "Fatal/Serious"]


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=RS)}


def build_schema(X):
    g = preprocessing.feature_groups(X)
    num_cols = g["numeric"] + g["skewed_numeric"]
    return {"feature_order": list(X.columns), "groups": {k: v for k, v in g.items() if k != "excluded"},
            "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())} for c in num_cols},
            "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in g["categorical"]},
            "excluded": g["excluded"], "target": config.TARGET}


def run_accident_modeling(df=None):
    df = data_loader.load_clean_accidents() if df is None else df
    X_tr, X_te, y_tr, y_te = preprocessing.make_split(df)
    # training-set metrics are measured on a (stratified-by-position) subset to keep KNN fast
    n_eval = min(config.TRAIN_EVAL_ROWS, len(X_tr))
    X_tr_eval, y_tr_eval = X_tr.iloc[:n_eval], y_tr.iloc[:n_eval]
    rows, details, fitted = [], {}, {}
    baseline_acc = float(max(y_te.mean(), 1 - y_te.mean()))
    for name, model in make_models().items():
        pipe = Pipeline([("prep", preprocessing.build_preprocessor(X_tr)), ("model", model)])
        pipe.fit(X_tr, y_tr)                       # preprocessing is fitted on TRAIN only
        p_te, p_tr = pipe.predict(X_te), pipe.predict(X_tr_eval)
        te, tr = evaluation.binary_metrics(y_te, p_te), evaluation.binary_metrics(y_tr_eval, p_tr)
        try:
            auc = float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
        except Exception:
            auc = float("nan")
        cm = evaluation.confusion(y_te, p_te, labels=[0, 1])
        evaluation.plot_confusion(cm, LABELS, f"{name}: confusion matrix (test)",
                                  config.FIG_DIR / f"cm_{name.lower().replace(' ', '_').replace('-', '_')}.png")
        gap = tr["f1"] - te["f1"]
        rows.append({"model": name, **{f"test_{k}": v for k, v in te.items()}, "test_roc_auc": auc,
                     "train_f1": tr["f1"], "train_accuracy": tr["accuracy"], "f1_gap_train_minus_test": gap,
                     "fit_diagnosis": "possible overfitting" if gap > 0.10 else ("possible underfitting" if te["f1"] < 0.25 else "no strong sign of over/underfitting")})
        details[name] = {"confusion_matrix": cm.tolist(), "test": te, "train": tr}
        fitted[name] = pipe
    table = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    table.to_csv(config.METRICS_DIR / "accident_model_comparison.csv", index=False)

    best = table.loc[0, "model"]
    justification = (f"'{best}' was selected because it has the highest F1-score for the Fatal/Serious class on held-out data. "
                     "Severe accidents are the minority class, so accuracy is misleading (always predicting 'Slight' would already "
                     f"score {baseline_acc:.3f}). F1 balances precision (avoiding false alarms) and recall (not missing severe cases). "
                     "If missing a severe case were considered worse than a false alarm, recall should be prioritised instead.")
    joblib.dump(fitted[best], config.MODEL_FILE)
    schema = build_schema(X_tr); data_loader.write_json(schema, config.SCHEMA_FILE)

    # model comparison chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = table.set_index("model")[["test_accuracy", "test_precision", "test_recall", "test_f1"]]
    t.plot.bar(ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Score (test set)"); ax.set_xlabel("Model")
    ax.set_title("Model comparison on held-out test data (precision/recall/F1 for Fatal/Serious class)")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout(); fig.savefig(config.FIG_DIR / "model_comparison.png", dpi=120); plt.close(fig)

    # random-forest importances (association with the prediction, NOT causation)
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5)); imp[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Random Forest importance"); ax.set_ylabel("Feature (after encoding)")
    ax.set_title("Top features used by the Random Forest (association, not causation)")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "rf_feature_importance.png", dpi=120); plt.close(fig)

    summary = {"target": "severe (1 = fatal or serious, 0 = slight)", "rows_used": int(len(X_tr) + len(X_te)),
               "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)), "test_severe_share": float(y_te.mean()),
               "majority_class_baseline_accuracy": baseline_acc, "primary_metric": "F1-score of the Fatal/Serious class",
               "best_model": best, "justification": justification, "features_used": list(X_tr.columns),
               "excluded_features": schema["excluded"], "models": details,
               "train_eval_rows": int(n_eval),
               "notes": "Preprocessing (imputation, scaling, encoding) is fitted inside each Pipeline on the training split only."}
    data_loader.write_json(summary, config.METRICS_DIR / "accident_models_summary.json")
    return table, summary
