"""Shared data-loading utilities used by main.py, the notebook and the Streamlit app."""
import json
import pandas as pd
import config

RENAME = {"collision_index": "accident_index", "collision_year": "accident_year",
          "collision_ref_no": "accident_reference", "collision_reference": "accident_reference",
          "collision_severity": "accident_severity"}

NO_DATA_MSG = ("No accident CSV found in {d}. Download the road-safety collision/accident file as described in "
               "data/README.md and place the CSV in that folder.")


def find_accident_csv():
    files = sorted(config.ACCIDENT_DIR.glob("*.csv"))
    return files[0] if files else None


def load_accidents(path=None):
    """Load the RAW accident CSV (never modified). Column names are normalised to lower_snake_case and the
    newer 'collision_*' DfT names are mapped to 'accident_*'."""
    path = path or find_accident_csv()
    if path is None:
        raise FileNotFoundError(NO_DATA_MSG.format(d=config.ACCIDENT_DIR))
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.rename(columns=RENAME)


def load_clean_accidents(path=None):
    path = path or config.CLEAN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py clean` first.")
    dtypes = {c: str for c in config.CATEGORICAL_FEATURES}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    for c in config.CATEGORICAL_FEATURES:
        if c in df:
            df[c] = df[c].astype(object)
    return df


def list_images(root=None):
    """Return [(path, label)] where label is the name of the sub-folder."""
    root = root or config.IMAGE_DIR
    items = []
    if not root.exists():
        return items
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in config.IMAGE_EXTENSIONS and p.parent != root:
            items.append((p, p.parent.name))
    return items


def find_match_pair():
    """Two related images of the same scene. Preference: data/raw/match_pair/; fallback: first two images of
    the image dataset (a warning is returned because they may not show the same scene)."""
    imgs = sorted(p for p in config.MATCH_PAIR_DIR.glob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if len(imgs) >= 2:
        return imgs[0], imgs[1], None
    ds = [p for p, _ in list_images()]
    if len(ds) >= 2:
        return ds[0], ds[1], ("No pair found in data/raw/match_pair/; used the first two dataset images, which may "
                              "NOT show the same scene. Matching results are then not meaningful.")
    raise FileNotFoundError("No images found. Put two related images in data/raw/match_pair/ and labelled images "
                            "in data/raw/images/<class_name>/.")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
