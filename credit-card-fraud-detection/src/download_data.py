"""Download creditcard.csv from Kaggle.  Fails clearly if credentials are absent."""
import sys
import subprocess
from pathlib import Path
from src.config import DATA_RAW, KAGGLE_DATASET, DATA_FILE


def download():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: kaggle package not installed. Run: pip install kaggle"
        )

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        sys.exit(
            f"ERROR: Kaggle credentials not found at {kaggle_json}.\n"
            "  1. Create a free account at https://www.kaggle.com\n"
            "  2. Go to Account → API → Create New Token\n"
            "  3. Move the downloaded kaggle.json to ~/.kaggle/\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)\n"
            "Then re-run this script."
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} to {DATA_RAW} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_RAW),
            "--unzip",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: Kaggle download failed.\n{result.stderr}")

    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: Expected {DATA_FILE} after download but it was not found.\n"
            "Check the Kaggle dataset structure."
        )
    print(f"Done. Data saved to {DATA_FILE}")


if __name__ == "__main__":
    download()
