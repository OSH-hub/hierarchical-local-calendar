from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = range(2016, 2024)
SEED = 7


def season_risk(rng: np.random.Generator, days: pd.DatetimeIndex, top: int) -> np.ndarray:
    doy = days.dayofyear.to_numpy()
    p = 0.5 * np.exp(-0.5 * ((doy - 205) / 30.0) ** 2)
    return np.where(rng.random(len(days)) < p, rng.choice([1, 2, top], len(days)), 0).astype(float)


def surveys(rng, county, year, start, rounds, base, shift, item, sido, code) -> list[dict]:
    rows, positive = [], False
    for k in range(rounds):
        day = dt.date(year, *start) + dt.timedelta(days=15 * k)
        doy = day.timetuple().tm_yday
        hazard = base * np.exp(-0.5 * ((doy - 200 - shift) / 22.0) ** 2)
        positive = positive or rng.random() < hazard
        rows.append({"examinYear": year, "examinTmrd": k + 1, "inputStdrDatetm": day.strftime("%Y%m%d"),
                     "insectKey": f"{year}{code}{k:02d}", "sidoCode": sido, "sigunguCode": code, "sigunguNm": county,
                     "dbyhsNm": item, "inqireCnClCode": "SX",
                     "inqireValue": round(rng.uniform(0.1, 5), 1) if positive else 0})
    return rows


def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    (out / "data" / "pairing").mkdir(parents=True, exist_ok=True)
    (out / "data" / "ncpms_api").mkdir(parents=True, exist_ok=True)
    apple_daily, rice_daily, apple_survey, rice_survey = [], [], [], []
    for i in range(12):
        county, sido, code = f"County{i:02d}", str(41 + i % 3), str(4100 + i)
        base, shift = rng.uniform(0.05, 0.6), rng.uniform(-20, 20)
        for year in YEARS:
            days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            for model in ("AMB", "ABR"):
                apple_daily.append(pd.DataFrame({"sigunguNm": county, "model": model,
                                                 "date": days.strftime("%Y-%m-%d"),
                                                 "risk_max": season_risk(rng, days, 3)}))
            rdays = pd.date_range(f"{year}-05-01", f"{year}-10-31")
            rice_daily.append(pd.DataFrame({"sidoCode": sido, "sigunguNm": county, "model": "RB_IR",
                                            "date": rdays.strftime("%Y-%m-%d"),
                                            "risk_max": season_risk(rng, rdays, 4)}))
            apple_survey += surveys(rng, county, year, (5, 15), 10, base, shift, "갈색무늬병(피해잎율)", sido, code)
            apple_survey += surveys(rng, county, year, (5, 15), 10, base * 0.6, shift + 15, "탄저병(피해과율)", sido, code)
            rice_survey += surveys(rng, county, year, (6, 1), 8, base * 0.5, shift, "잎도열병(발생면적율)", sido, code)
    pd.concat(apple_daily).to_csv(out / "data/pairing/pairing_daily.csv", index=False)
    pd.concat(rice_daily).to_csv(out / "data/pairing/pairing_daily_rice.csv", index=False)
    pd.DataFrame(apple_survey).to_csv(out / "data/ncpms_api/apple_surveillance_sigungu.csv", index=False)
    pd.DataFrame(rice_survey).to_csv(out / "data/ncpms_api/rice_surveillance_sigungu.csv", index=False)
    print("synthetic inputs written to", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
