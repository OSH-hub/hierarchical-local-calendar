from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA256 = {
    "data/pairing/pairing_daily.csv": "7e436575d67fceab6f8c2884a3ca5168adef79fc479cbfd5079e5a32351fa207",
    "data/pairing/pairing_daily_rice.csv": "dad24668e0cdd1696c0e2f7057bf4d55da8e83e0f8701664d292d9f072d86fbc",
    "data/ncpms_api/apple_surveillance_sigungu.csv": "bc92a3689b43afd15cc8d699ec0f4599977720425f317754abeb311b09a63654",
    "data/ncpms_api/rice_surveillance_sigungu.csv": "6ed10e859c6da3099ebb64fcf2c76109dda908749f017c125dbbb1c8a5506f78",
}
APPLE_ITEMS = {"갈색무늬병(피해잎율)": "AMB", "탄저병(피해과율)": "ABR"}
RICE_ITEM = "잎도열병(발생면적율)"
MODELS = ("AMB", "ABR", "RB_IR")
LAST_YEAR = 2025


@dataclass
class Record:
    model: str
    county: str
    year: int
    province: str
    days: np.ndarray
    risk: np.ndarray
    obs: np.ndarray
    values: np.ndarray


RICE_DAILY = "data/pairing/pairing_daily_rice.csv"


def _read(root: Path, rel: str, override: Path | None = None, check_hash: bool = True, **kw) -> pd.DataFrame:
    path = Path(override) if override else root / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if check_hash and digest != EXPECTED_SHA256[rel]:
        raise ValueError(f"input hash differs from the frozen source: {rel}")
    return pd.read_csv(path, **kw)


def load_records(root: Path, rice_daily: Path | None = None, check_hash: bool = True) -> tuple[dict[tuple[str, str, int], Record], dict]:
    root = Path(root)
    ad = _read(root, "data/pairing/pairing_daily.csv", check_hash=check_hash)
    rd = _read(root, RICE_DAILY, rice_daily, check_hash, dtype={"sidoCode": str})
    opts = {"check_hash": check_hash, "dtype": {"inputStdrDatetm": str, "sidoCode": str, "sigunguCode": str}}
    asu = _read(root, "data/ncpms_api/apple_surveillance_sigungu.csv", **opts)
    rsu = _read(root, "data/ncpms_api/rice_surveillance_sigungu.csv", **opts)
    flow = {"apple_raw_rows": len(asu), "rice_raw_rows": len(rsu)}

    ad["county"] = ad.sigunguNm
    rd["county"] = rd.sidoCode + "|" + rd.sigunguNm
    asu = asu[asu.dbyhsNm.isin(APPLE_ITEMS)].copy()
    asu["model"] = asu.dbyhsNm.map(APPLE_ITEMS)
    asu["county"] = asu.sigunguNm
    rsu = rsu[rsu.dbyhsNm == RICE_ITEM].copy()
    rsu["model"] = "RB_IR"
    rsu["county"] = rsu.sidoCode + "|" + rsu.sigunguNm
    flow["apple_item_rows"] = len(asu)
    flow["rice_item_rows"] = len(rsu)

    daily = pd.concat([ad, rd], ignore_index=True)
    daily["year"] = daily.date.str[:4].astype(int)
    daily = daily[daily.year <= LAST_YEAR].copy()
    daily["day"] = pd.to_datetime(daily.date).map(dt.date.toordinal)

    surv = pd.concat([asu, rsu], ignore_index=True)
    surv["year"] = surv.inputStdrDatetm.str[:4].astype(int)
    surv["value"] = pd.to_numeric(surv.inqireValue, errors="coerce")
    flow["item_rows_through_2025"] = surv[surv.year <= LAST_YEAR].groupby("model").size().to_dict()
    surv = surv[(surv.year <= LAST_YEAR) & surv.value.notna()].copy()
    flow["numeric_rows"] = surv.groupby("model").size().to_dict()
    surv["day"] = pd.to_datetime(surv.inputStdrDatetm, format="%Y%m%d").map(dt.date.toordinal)
    if daily.duplicated(["model", "county", "day"]).any() or surv.duplicated(["model", "county", "day"]).any():
        raise ValueError("duplicate model-county-day rows")

    keys = set(map(tuple, daily[["model", "county"]].drop_duplicates().to_numpy()))
    surv = surv[[tuple(v) in keys for v in surv[["model", "county"]].to_numpy()]].copy()
    flow["rows_in_counties_with_risk"] = surv.groupby("model").size().to_dict()
    flow["survey_counties_total"] = {m: int(g.county.nunique()) for m, g in pd.concat([asu, rsu]).groupby("model")}
    flow["risk_counties"] = daily.groupby("model").county.nunique().to_dict()

    province = surv.groupby(["model", "county"]).sidoCode.first().to_dict()
    surveys = {k: (g.day.to_numpy(), g.value.to_numpy())
               for k, g in surv.sort_values("day").groupby(["model", "county", "year"])}
    records = {}
    for (m, c, y), g in daily.sort_values("day").groupby(["model", "county", "year"]):
        obs, vals = surveys.get((m, c, y), (np.array([], dtype=int), np.array([])))
        records[m, c, int(y)] = Record(m, c, int(y), province.get((m, c), "unavailable"),
                                       g.day.to_numpy(), g.risk_max.to_numpy(), obs, vals)
    return records, flow
