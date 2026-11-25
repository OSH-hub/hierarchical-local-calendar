"""Aggregate descriptions of the input data, used for the data figures.

Usage
-----
python src/run_data_summaries.py --project <project root> [--rice-daily <file>] [--out <dir>]

Writes
  tables/D01_county_coordinates.csv      centre of the three grid cells of each county
  tables/D02_first_detection_matrix.csv  per county-year: number of valid surveys and day of first detection
  tables/D03_warning_share_by_day.csv    share of counties at or above each risk threshold, by day of year and year
  tables/D04_survey_days.csv             day of year of every valid survey and whether it was positive
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from loader import MODELS, load_records
from run_reanalysis import THRESHOLDS


def doy(ordinal: int) -> int:
    return dt.date.fromordinal(int(ordinal)).timetuple().tm_yday


def coordinates(root: Path) -> pd.DataFrame:
    apple = pd.read_csv(root / "data/agri_weather/valid_cells.csv", dtype={"sidoCode": str})
    rice = pd.read_csv(root / "data/agri_weather/valid_cells_rice.csv", dtype={"sidoCode": str})
    apple["county"] = apple["sigunguNm"]
    rice["county"] = rice["sidoCode"] + "|" + rice["sigunguNm"]
    frames = []
    for crop, frame in (("apple", apple), ("rice", rice)):
        g = frame.groupby("county")[["lat", "lon"]].mean().reset_index()
        g.insert(0, "crop", crop)
        frames.append(g)
    return pd.concat(frames).round({"lat": 4, "lon": 4})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--rice-daily", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    records, _ = load_records(args.project, args.rice_daily)
    tables = args.out / "tables"

    coordinates(args.project).to_csv(tables / "D01_county_coordinates.csv", index=False)

    matrix, days, share = [], [], []
    for model in MODELS:
        recs = [r for (m, _, _), r in records.items() if m == model]
        for r in recs:
            positive = r.obs[r.values > 0]
            matrix.append({"model": model, "county": r.county, "province": r.province, "year": r.year,
                           "valid_surveys": len(r.obs),
                           "first_detection_doy": doy(positive.min()) if len(positive) else np.nan})
            days += [{"model": model, "year": r.year, "doy": doy(o), "positive": bool(v > 0)}
                     for o, v in zip(r.obs, r.values)]
        for year in sorted({r.year for r in recs}):
            grid = np.full((sum(r.year == year for r in recs), 367), np.nan)
            for i, r in enumerate([r for r in recs if r.year == year]):
                grid[i, [doy(x) for x in r.days]] = r.risk
            for t in THRESHOLDS[model]:
                with np.errstate(invalid="ignore"):
                    frac = np.nanmean(grid >= t, axis=0)
                frac[np.isnan(grid).all(axis=0)] = np.nan
                share.append(pd.DataFrame({"model": model, "year": year, "threshold": t, "doy": np.arange(367),
                                           "share_of_counties": frac}).dropna())
    pd.DataFrame(matrix).to_csv(tables / "D02_first_detection_matrix.csv", index=False)
    pd.concat(share).round({"share_of_counties": 4}).to_csv(tables / "D03_warning_share_by_day.csv", index=False)
    pd.DataFrame(days).to_csv(tables / "D04_survey_days.csv", index=False)


if __name__ == "__main__":
    main()
