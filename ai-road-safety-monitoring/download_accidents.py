"""Download the official UK DfT 2025 road-collision CSV into the expected project folder.

Run from the repository root:
    python download_accidents.py

Source: UK Department for Transport Road Safety Open Data.
This script downloads only the collision table; image datasets are separate and still need
to be added for the computer-vision sections.
"""
from pathlib import Path
from urllib.request import urlopen
import shutil
import sys

URL = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2025.csv"
DEST = Path(__file__).resolve().parent / "data" / "raw" / "accidents" / "dft-road-casualty-statistics-collision-2025.csv"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading official UK Department for Transport collision data (2025)...")
    print("Source:", URL)
    try:
        with urlopen(URL, timeout=60) as response, DEST.open("wb") as out:
            shutil.copyfileobj(response, out)
        if DEST.stat().st_size < 100_000:
            DEST.unlink(missing_ok=True)
            raise RuntimeError("Downloaded file is unexpectedly small; check the source URL.")
        print(f"Saved: {DEST}")
        print(f"Size: {DEST.stat().st_size / (1024*1024):.1f} MB")
        print("Next: run `python main.py clean`, then `python main.py eda`, `python main.py model`.")
        print("Note: computer-vision stages require images in data/raw/images and data/raw/match_pair.")
    except Exception as exc:
        print("Download failed:", exc)
        print("Open the official source page and download the 2025 collision CSV manually:")
        print("https://www.gov.uk/government/statistical-data-sets/road-safety-open-data")
        sys.exit(1)

if __name__ == "__main__":
    main()
