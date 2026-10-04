"""Normalize and sort the didinium/paramecium observations for example_3.py."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/didinium_paramecium.csv"
DATA_PATH = ROOT / "data/didinium_paramecium_div100.csv"


def main() -> None:
    with RAW_PATH.open() as file:
        rows = sorted(csv.DictReader(file), key=lambda row: float(row["time"]))

    with DATA_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("time", "paramecium", "didinium"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time": row["time"],
                    "paramecium": float(row["paramecium"]) / 100.0,
                    "didinium": float(row["didinium"]) / 100.0,
                }
            )
    print(f"saved: {DATA_PATH}")


if __name__ == "__main__":
    main()
