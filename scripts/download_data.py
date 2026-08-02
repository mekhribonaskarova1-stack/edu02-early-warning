"""
Downloads the real OULAD dataset and unzips the CSVs into data/raw/.
Source: https://analyse.kmi.open.ac.uk/open_dataset (CC-BY 4.0)

Run: python scripts/download_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OULAD from {DATA_URL} ...")
    try:
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(RAW_DIR)
        print(f"Done. CSVs extracted to {RAW_DIR}")
    except Exception as exc:
        print(
            f"Automatic download failed ({exc}).\n"
            f"Please download the dataset manually from {DATA_URL.rsplit('/download', 1)[0]} "
            f"and place the CSV files in {RAW_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()