#!/usr/bin/env python
"""Build the ChemCrow-adapted RetroBioCat 52-pathway benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.benchmarks.retrobiocat52 import build_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT.parent
            / "dataset"
            / "RetroBioCat 52-pathway test set.xlsx"
        ),
        help="source RetroBioCat XLSX",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset" / "retrobiocat52",
        help="directory for benchmark.jsonl and manifest.json",
    )
    parser.add_argument(
        "--sheet", default=None, help="worksheet name; defaults to first"
    )
    args = parser.parse_args()

    manifest = build_benchmark(args.input, args.output_dir, sheet_name=args.sheet)
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))
    print(f"Wrote {manifest['case_count']} cases to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
