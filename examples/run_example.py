from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import reference as ref  # noqa: E402
import verification as ver  # noqa: E402
from loader import load_records  # noqa: E402
from run_local_model import LEVELS, encode, fit_on, predict_on, rolling_folds  # noqa: E402


def main(root: Path) -> dict:
    records, _ = load_records(root, check_hash=False)
    out = {}
    for model in ("AMB", "RB_IR"):
        units, _, _ = ver.build_units(records, model, window=45)
        units = units[units["at_risk"]].reset_index(drop=True)
        d = encode(units)
        pred = {k: np.full(len(units), np.nan) for k in ("national_calendar", "county_calendar")}
        for train, test in rolling_folds(d):
            fitted = fit_on(d, train)
            for k in pred:
                pred[k][test] = predict_on(fitted, d, test, LEVELS[k])
        ok = ~np.isnan(pred["county_calendar"])
        obs = d["pos"][ok]
        out[model] = {"units": int(ok.sum()), "events": int(obs.sum()),
                      "AUC_operational": round(ref.auc(units["score"].to_numpy()[ok], obs), 3),
                      **{f"AUC_{k}": round(ref.auc(v[ok], obs), 3) for k, v in pred.items()}}
        print(model, out[model])
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
