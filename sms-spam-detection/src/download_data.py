"""Download the SMS Spam Collection dataset from Kaggle.

Usage:
    python -m src.download_data

Requires Kaggle API credentials in ~/.kaggle/kaggle.json (or
KAGGLE_USERNAME / KAGGLE_KEY environment variables).
If credentials are missing this script fails with a clear message.
"""
import sys
from pathlib import Path

from src.config import DATA_RAW, KAGGLE_DATASET


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Check credentials before importing kaggle so we give a helpful error
    import os
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_creds = (
        kaggle_json.exists()
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    if not has_creds:
        print(
            "ERROR: Kaggle credentials not found.\n"
            "Either:\n"
            "  1. Place your kaggle.json in ~/.kaggle/kaggle.json  (chmod 600)\n"
            "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n\n"
            "Get your API key at https://www.kaggle.com/settings → API → Create New Token\n\n"
            "Alternatively, download spam.csv manually from:\n"
            f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and place it in: {DATA_RAW}/",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import kaggle  # noqa: F401 — imported after credentials check
    except ImportError:
        print(
            "ERROR: kaggle package not installed. Run: pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading {KAGGLE_DATASET} to {DATA_RAW} ...")
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_RAW),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: kaggle download failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    expected = DATA_RAW / "spam.csv"
    if not expected.exists():
        # Dataset might unzip to a subfolder; look for it
        candidates = list(DATA_RAW.rglob("spam.csv"))
        if candidates:
            import shutil
            shutil.copy(candidates[0], expected)
            print(f"  Moved {candidates[0]} → {expected}")
        else:
            print(
                f"ERROR: spam.csv not found under {DATA_RAW} after download.\n"
                "Please place it there manually.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Done. Data available at: {expected}")


if __name__ == "__main__":
    main()
