"""Command-line, read-only inspection of files under data/raw."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.loader import inspect_raw_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Inspect raw dataset files without modifying them."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="Raw data directory (default: repository data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    """Print a JSON dataset inspection report."""

    args = parse_args()
    report = inspect_raw_dataset(args.raw_dir)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
