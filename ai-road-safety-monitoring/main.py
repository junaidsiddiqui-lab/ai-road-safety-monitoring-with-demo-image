"""Command-line entry point.  Usage:  python main.py [check|clean|eda|images|match|classify|model|report|all]"""
import sys
import traceback
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import config
from src import data_loader as dl, data_cleaning, eda, image_processing, feature_detection, image_classification, accident_modeling, reporting

STAGES = ["check", "clean", "eda", "images", "match", "classify", "model", "report"]


def stage_check():
    csv = dl.find_accident_csv(); imgs = dl.list_images()
    print("Accident CSV:", csv or "MISSING (see data/README.md)")
    print("Labelled images:", len(imgs), "| classes:", sorted({l for _, l in imgs}) or "none")
    try:
        a, b, w = dl.find_match_pair(); print("Matching pair:", a.name, b.name, "| warning:", w)
    except FileNotFoundError as e:
        print("Matching pair: MISSING -", e)


def stage_clean():
    df, summary = data_cleaning.clean_accidents(dl.load_accidents())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    from src import preprocessing
    summary["feature_types"] = preprocessing.feature_groups(df)
    dl.write_json(summary, config.CLEANING_SUMMARY)
    print(f"Raw {summary['raw']['rows']:,} rows -> clean {len(df):,} rows; severe share {summary['severe_share']:.1%}")


def stage_eda():
    findings, _ = eda.run_eda(dl.load_clean_accidents())
    print(*findings, sep="\n")


def stage_images():
    imgs = [p for p, _ in dl.list_images()][:3]
    a, b, _ = dl.find_match_pair()
    paths = list(dict.fromkeys([a, b] + imgs))
    res = {"images": [image_processing.process_image(p) for p in paths]}
    dl.write_json(res, config.METRICS_DIR / "image_processing.json")
    print(f"Processed {len(paths)} images")


def stage_match():
    a, b, warn = dl.find_match_pair()
    res = feature_detection.analyse_pair(a, b)
    res["warning"] = warn
    dl.write_json(res, config.METRICS_DIR / "feature_matching.json")
    print({k: res[k] for k in ("crosscheck", "ratio")})


def stage_classify():
    res = image_classification.run_image_classification()
    print(res.get("status"), res.get("reason", ""))


def stage_model():
    table, summary = accident_modeling.run_accident_modeling()
    print(table[["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "fit_diagnosis"]].to_string(index=False))
    print("Best:", summary["best_model"])


def stage_report():
    print("Report written to", reporting.build_report())


FUNCS = {"check": stage_check, "clean": stage_clean, "eda": stage_eda, "images": stage_images, "match": stage_match,
         "classify": stage_classify, "model": stage_model, "report": stage_report}


def run_stage(name):
    config.ensure_dirs()
    status = dl.read_json(config.STATUS_FILE, {})
    try:
        FUNCS[name]()
        status[name] = "ok"
    except FileNotFoundError as e:
        status[name] = f"skipped: {e}"; print(f"[{name}] SKIPPED: {e}")
    except Exception as e:  # keep going so other stages can still run
        status[name] = f"failed: {type(e).__name__}: {e}"; traceback.print_exc()
    dl.write_json(status, config.STATUS_FILE)
    return status[name]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg not in STAGES + ["all"]:
        sys.exit(f"Unknown stage '{arg}'. Choose from: {', '.join(STAGES + ['all'])}")
    for s in (STAGES if arg == "all" else [arg]):
        print(f"\n=== {s} ===")
        run_stage(s)
