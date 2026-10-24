"""Download the IBM Telco Customer Churn CSV into data/raw/ (skipped if it already exists)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_PATH  # noqa: E402
from src.data_loader import DataLoadError, download_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args()
    try:
        path = download_dataset(force=args.force)
    except DataLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset ready at {path}" if path == RAW_PATH else f"Dataset ready at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
