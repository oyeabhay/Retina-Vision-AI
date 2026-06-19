"""
download_dataset.py
───────────────────
Downloads the OCT2017 dataset from Kaggle.

Prerequisites
─────────────
1. Install the Kaggle CLI:  pip install kaggle
2. Place your API token at  ~/.kaggle/kaggle.json
   (Create one at: https://www.kaggle.com/account → API → Create New Token)

Usage
─────
    python download_dataset.py
or
    python download_dataset.py --dest /custom/path
"""

import os
import argparse
import subprocess
import sys
import zipfile


KAGGLE_DATASET = "anirudhcv/labeled-optical-coherence-tomography-oct"
DEFAULT_DEST    = "."


def check_kaggle_cli():
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download(dest: str):
    os.makedirs(dest, exist_ok=True)

    if not check_kaggle_cli():
        print("❌  Kaggle CLI not found. Install it with:\n    pip install kaggle")
        print("\nAlternatively, download manually:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        sys.exit(1)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌  Kaggle API credentials not found.")
        print("    Create a token at: https://www.kaggle.com/account → API")
        print(f"    Then place it at:   {kaggle_json}")
        sys.exit(1)

    print(f"⬇️  Downloading {KAGGLE_DATASET} to {dest} …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", dest],
        check=True,
    )

    # Find the downloaded zip
    zip_name = KAGGLE_DATASET.split("/")[-1] + ".zip"
    zip_path = os.path.join(dest, zip_name)

    if os.path.exists(zip_path):
        print(f"📦  Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        os.remove(zip_path)
        print(f"✅  Dataset ready at {dest}/OCT2017/")
    else:
        print("⚠️  Zip file not found; Kaggle may have extracted it automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OCT2017 dataset from Kaggle")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST,
                        help="Directory to save the dataset")
    args = parser.parse_args()
    download(args.dest)
