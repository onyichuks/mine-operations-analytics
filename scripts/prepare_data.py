"""Convert the flotation file to numbers and extract the gold mine records from the MSHA files.

Usage:  python scripts/prepare_data.py [--data data]     (about two minutes; the violations file is streamed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mineops.data import prepare_flotation, prepare_gold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    data = Path(args.data)
    out = prepare_flotation(data / "raw", data)
    print(f"Flotation plant data written to {out}")
    counts = prepare_gold(data / "raw", data)
    print("Gold mine records:", ", ".join(f"{k} {v:,}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
