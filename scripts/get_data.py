"""Download the public datasets into data/raw and check them.

Usage:  python scripts/get_data.py [--data data] [--strict]

The flotation file on OpenML is fixed, so its SHA-256 must match. MSHA refreshes its files every Friday, so a
fresh MSHA download will usually differ from the snapshot used for the published results (27 October 2026);
the script reports this and carries on unless --strict is given. About 330 MB in total.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mineops.data import SNAPSHOT_DATE, SOURCES, sha256  # noqa: E402


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mine-operations-analytics (research)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.0f} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--strict", action="store_true", help="fail if an MSHA file differs from the published snapshot")
    args = ap.parse_args()
    raw = Path(args.data) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    problems = []
    for key, (url, name, expected) in SOURCES.items():
        dest = raw / name
        if not dest.exists():
            print(f"Downloading {name}")
            download(url, dest)
        got = sha256(dest)
        if got == expected:
            print(f"  {name}: checksum matches")
        elif key == "flotation":
            problems.append(f"{name}: checksum {got} does not match {expected}")
        else:
            msg = f"{name}: newer than the {SNAPSHOT_DATE} snapshot (MSHA updates weekly); results will differ slightly"
            print("  " + msg)
            if args.strict:
                problems.append(msg)
    if problems:
        sys.exit("\n".join(problems))
    print("Done. Next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
