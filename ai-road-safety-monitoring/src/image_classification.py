"""Traditional image classifier: ORB descriptors -> Bag of Visual Words histogram -> SVM / k-NN.
The vocabulary (k-means) and classifiers are learned from TRAINING images only. Images from the same source
scene are kept in the same split (group-aware split) to reduce leakage from near-duplicates."""
import re
import cv2
import joblib
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import config
from src import data_loader, evaluation, image_processing as ip, feature_detection as fd


def group_id(path):
    return re.split(config.GROUP_SPLIT_REGEX, path.stem)[0]


def extract_descriptors(path):
    g = ip.apply_clahe(ip.to_gray(ip.resize_max(ip.load_image(path), config.CLASSIFY_MAX_SIDE)))
    _, des = fd.detect_orb(g, config.CLASSIFY_ORB_FEATURES)
    return None if des is None else np.unpackbits(des, axis=1).astype(np.float32)  # 256 bits per descriptor


def bovw_histogram(des, kmeans):
    k = kmeans.n_clusters
    if des is None or len(des) == 0:
        return np.zeros(k, dtype=np.float32)
    h = np.bincount(kmeans.predict(des), minlength=k).astype(np.float32)
    return h / h.sum()


def run_image_classification():
    items = data_loader.list_images()
    counts = Counter(l for _, l in items)
    usable = {l for l, c in counts.items() if c >= config.MIN_IMAGES_PER_CLASS}
    if len(usable) < 2:
        res = {"status": "skipped", "class_counts": dict(counts),
               "reason": f"Need at least 2 classes with >= {config.MIN_IMAGES_PER_CLASS} images each in data/raw/images/<class>/."}
        data_loader.write_json(res, config.METRICS_DIR / "image_classification.json")
        return res
    items = [(p, l) for p, l in items if l in usable]
    paths = [p for p, _ in items]; y = np.array([l for _, l in items]); groups = np.array([group_id(p) for p in paths])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    tr, te = next(sgkf.split(paths, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "group leakage between train and test"
    des = [extract_descriptors(p) for p in paths]
    train_des = np.vstack([d for i, d in enumerate(des) if i in set(tr) and d is not None])
    rng = np.random.default_rng(config.RANDOM_STATE)
    if len(train_des) > 100000:
        train_des = train_des[rng.choice(len(train_des), 100000, replace=False)]
    kmeans = MiniBatchKMeans(n_clusters=config.VOCAB_SIZE, random_state=config.RANDOM_STATE, n_init=3, batch_size=1024).fit(train_des)
    H = np.vstack([bovw_histogram(d, kmeans) for d in des])
    models = {"SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", class_weight="balanced", random_state=config.RANDOM_STATE)),
              "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))}
    classes = sorted(usable); out = {}
    for name, m in models.items():
        m.fit(H[tr], y[tr]); pred = m.predict(H[te])
        cm = evaluation.confusion(y[te], pred, labels=classes)
        evaluation.plot_confusion(cm, classes, f"{name}: image confusion matrix (test)",
                                  config.FIG_DIR / f"img_cm_{name.split()[0].lower().replace('-', '')}.png")
        out[name] = {**evaluation.macro_metrics(y[te], pred), "confusion_matrix": cm.tolist(),
                     "report": classification_report(y[te], pred, labels=classes, output_dict=True, zero_division=0)}
    best = max(out, key=lambda k: out[k]["f1_macro"])
    joblib.dump({"kmeans": kmeans, "classifier": models[best], "classes": classes}, config.MODEL_DIR / "image_bovw_classifier.joblib")
    res = {"status": "done", "classes": classes, "class_counts": dict(counts), "n_images": len(paths),
           "n_groups": int(len(set(groups))), "train_images": int(len(tr)), "test_images": int(len(te)),
           "test_class_counts": dict(Counter(y[te])), "vocabulary_size": config.VOCAB_SIZE,
           "descriptor_bits": 256, "models": out, "best_model_by_macro_f1": best,
           "note": "Train/test split is group-aware; vocabulary and classifier were fitted on training images only."}
    data_loader.write_json(res, config.METRICS_DIR / "image_classification.json")
    return res
