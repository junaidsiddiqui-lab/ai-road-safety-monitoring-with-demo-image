"""Builds outputs/reports/final_report.md ONLY from result files that exist. Missing results are reported as
'not run', never invented."""
import pandas as pd
import config
from src import data_loader as dl

NOT_RUN = "_Not available: this stage has not been run (or was skipped). Run `python main.py all` after adding the data._"


def _fig(name, caption):
    return f"![{caption}](../figures/{name})\n\n*{caption}*\n" if (config.FIG_DIR / name).exists() else ""


def _table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def build_report():
    clean = dl.read_json(config.CLEANING_SUMMARY)
    eda = dl.read_json(config.EDA_FINDINGS)
    proc = dl.read_json(config.METRICS_DIR / "image_processing.json")
    feat = dl.read_json(config.METRICS_DIR / "feature_matching.json")
    cls = dl.read_json(config.METRICS_DIR / "image_classification.json")
    mod = dl.read_json(config.METRICS_DIR / "accident_models_summary.json")
    status = dl.read_json(config.STATUS_FILE, {})
    L = [f"# {config.PROJECT_TITLE}", "", f"**Student:** {config.STUDENT_NAME}  \n**Roll number:** {config.ROLL_NUMBER}  \n**University:** {config.UNIVERSITY}", "",
         "## 1. Problem and objectives",
         "This educational project analyses real road-accident records and real road/motorcycle images in one workflow: (A) accident analytics and severity prediction, (B) classical computer-vision analysis of images. The two components share the road-safety objective but use **separate datasets**: no image is linked to any accident record, and no claim of such a link is made.", "",
         "## 2. Data", "Dataset names, links and licences are listed in `LICENSE_AND_DATA_SOURCES.md`."]
    if clean:
        r, c = clean["raw"], clean["clean"]
        L += [f"- Raw accident file: **{r['rows']:,} rows x {r['columns']} columns**. After cleaning: **{c['rows']:,} rows x {c['columns']} columns**.",
              f"- Target `severe` (fatal or serious = 1): {clean['target_counts']} ({clean['severe_share']:.1%} severe)."]
    else:
        L.append(NOT_RUN)
    L += ["", "## 3. Data cleaning decisions"]
    if clean:
        L += ["| Problem found | Method applied | Why |", "|---|---|---|"] + [f"| {s['problem']} | {s['method']} | {s['why']} |" for s in clean["steps"]]
    else:
        L.append(NOT_RUN)
    L += ["", "## 4. Exploratory data analysis"]
    if eda:
        L += [f"{i}. {f}" for i, f in enumerate(eda["findings"], 1)]
        L += ["", "Associations are observational and do not prove causation.", ""]
        for key, cap in [("target_distribution", "Target distribution"), ("hourly", "Accidents and severe share by hour"),
                         ("feature_vs_target", "Severe share by feature"), ("correlation", "Correlation heatmap"), ("multivariate", "Time band x speed limit")]:
            if key in eda["figures"]:
                L.append(_fig(eda["figures"][key], cap))
    else:
        L.append(NOT_RUN)
    L += ["", "## 5. Image processing"]
    if proc:
        L += [f"- {p['image']}: {p['width']}x{p['height']} px, {p['channels']} channels, dtype {p['dtype']}; grey std before/after CLAHE {p['gray_std_before_clahe']:.1f} -> {p['gray_std_after_clahe']:.1f}." for p in proc["images"]]
        L += [_fig(p["comparison_figure"], f"Processing stages for {p['image']}") for p in proc["images"][:2]]
    else:
        L.append(NOT_RUN)
    L += ["", "## 6. Feature detection and matching"]
    if feat:
        for p in feat["per_image"]:
            L.append(f"- {p['image']}: {p['orb_keypoints']} ORB keypoints, descriptor array {p['descriptor_shape']} ({p['descriptor_bits']} bits), Canny thresholds {p['canny_thresholds']}.")
        for m in ("crosscheck", "ratio"):
            x = feat[m]; L.append(f"- {m} matching: {x['matches']} matches, {x['ransac_inliers']} RANSAC-consistent inliers" + (f" ({x['inlier_share']:.0%})" if x["inlier_share"] is not None else "") + ".")
        if feat.get("warning"):
            L.append(f"\n**Warning:** {feat['warning']}")
        L += ["", _fig("edges_and_keypoints.png", "Canny edges and ORB keypoints")]
        L.append("Limitations: ORB is sensitive to large viewpoint, scale and lighting changes and to blur; ratio-test matches can still be wrong, which is why RANSAC inliers are also reported.")
    else:
        L.append(NOT_RUN)
    L += ["", "## 7. Image classification (ORB Bag of Visual Words + SVM/k-NN)"]
    if cls and cls.get("status") == "done":
        L += [f"- Classes: {cls['classes']}; {cls['n_images']} images from {cls['n_groups']} source groups; train {cls['train_images']} / test {cls['test_images']} (group-aware split)."]
        L += ["", "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |", "|---|---|---|---|---|"]
        L += [f"| {n} | {m['accuracy']:.3f} | {m['precision_macro']:.3f} | {m['recall_macro']:.3f} | {m['f1_macro']:.3f} |" for n, m in cls["models"].items()]
    elif cls:
        L.append(f"Classification was **skipped**: {cls.get('reason')}")
    else:
        L.append(NOT_RUN)
    L += ["", "## 8. Accident machine learning"]
    if mod:
        L += [f"Prediction problem: {mod['target']} using only information available at the time of the accident report (no casualty counts, no police-attendance flag). {mod['train_rows']:,} training / {mod['test_rows']:,} test rows (stratified, random_state={config.RANDOM_STATE}). Majority-class baseline accuracy: {mod['majority_class_baseline_accuracy']:.3f}.", ""]
        t = pd.read_csv(config.METRICS_DIR / "accident_model_comparison.csv")
        L += [_table(t[["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "train_f1", "fit_diagnosis"]]), "",
              f"**Best model:** {mod['best_model']}. {mod['justification']}", "", _fig("model_comparison.png", "Model comparison"), _fig("rf_feature_importance.png", "Random Forest feature importance")]
    else:
        L.append(NOT_RUN)
    L += ["", "## 9. Limitations",
          "- Accident records are observational: associations (and model feature importances) are **not** causal.",
          "- Predictions are educational and do not guarantee, measure or improve actual road safety.",
          "- Under-reporting of slight accidents, coded categories and one national/regional source limit generalisation.",
          "- Image results depend on the chosen image dataset; classical ORB features are weaker than modern deep learning.",
          "- The system does **not** detect helmet violations or analyse live video; no object-detection model was trained.", "",
          "## 10. Run status", "```", str(status), "```", "", "## 11. Conclusion",
          "The project provides a reproducible pipeline combining cleaning, EDA, classical image analysis and tabular machine learning under one road-safety theme. Conclusions are limited to the numbers shown above."]
    out = config.REPORT_DIR / "final_report.md"
    out.write_text("\n".join(L))
    return out
