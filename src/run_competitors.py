"""Compare the hierarchical local calendar with competing forecasting models.

Every competitor uses the surveillance record only, is refitted in each rolling-origin
fold on the years before the test year, and is scored on the same at-risk units.
Settings of all competitors were fixed before evaluation.

Competitors
  county_unpooled       the county calendar without partial pooling (prior strength 1e-6 events)
  persistence           the county's most recent first-detection day before the test year,
                        smoothed with a Gaussian of 15 days (one survey interval); 0 if not detected
  logistic_mixed        logistic regression on a cubic B-spline of day of year (8 knots) with
                        province and county effects shrunk toward zero (ridge penalty, C = 1),
                        the penalized form of a random-intercept model
  random_forest         500 trees, minimum leaf 20 units
  gradient_boosting     histogram gradient boosting, 300 iterations, learning rate 0.05,
                        15 leaves, minimum leaf 20 units
The two tree models use day of year and county history from the years before each unit's
year: share of surveyed years with a detection, mean first-detection day, years of history,
detection in the most recent year and its day, and the same share for the province.

Usage
-----
python src/run_competitors.py [--out <dir>]

Reads results/verification_units.csv (run_reanalysis.py).
Writes tables/M15_competing_models.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import local_calendar as lc
import reference as ref
import verification as ver
from run_local_model import encode, fit_on, predict_on, rolling_folds, brier, LEVELS
from run_reanalysis import N_BOOT

PERSIST_BANDWIDTH = 15.0
FEATURES = ["doy", "c_rate", "c_mean_doy", "c_years", "c_last_detected", "c_last_doy", "p_rate"]


def history_features(units: pd.DataFrame) -> pd.DataFrame:
    """County and province history from the years strictly before each unit's year."""
    cy = (units.groupby(["county", "province", "year"])
          .apply(lambda g: pd.Series({"det": bool(g["positive"].any()),
                                      "fd": g.loc[g["positive"], "doy"].min() if g["positive"].any() else np.nan}),
                 include_groups=False)
          .reset_index())
    province_of = units.groupby("county")["province"].first()
    rows = []
    for (c, y), _ in units.groupby(["county", "year"]):
        past = cy[(cy["county"] == c) & (cy["year"] < y)].sort_values("year")
        prov = cy[(cy["province"] == province_of[c]) & (cy["year"] < y)]
        nat_mean = cy.loc[cy["year"] < y, "fd"].mean()
        rate = past["det"].mean() if len(past) else np.nan
        p_rate = prov["det"].mean() if len(prov) else np.nan
        mean_fd = past["fd"].mean() if past["det"].any() else (prov["fd"].mean() if prov["det"].any() else nat_mean)
        last = past.iloc[-1] if len(past) else None
        rows.append({"county": c, "year": y, "c_rate": rate, "c_mean_doy": mean_fd, "c_years": len(past),
                     "c_last_detected": float(last["det"]) if last is not None else np.nan,
                     "c_last_doy": last["fd"] if last is not None and last["det"] else np.nan, "p_rate": p_rate})
    return units.merge(pd.DataFrame(rows), on=["county", "year"], how="left")


def persistence(feat: pd.DataFrame, test: np.ndarray) -> np.ndarray:
    last = feat["c_last_doy"].to_numpy()[test]
    doy = feat["doy"].to_numpy()[test]
    s = np.exp(-0.5 * ((doy - last) / PERSIST_BANDWIDTH) ** 2)
    return np.where(np.isnan(last), 0.0, s)


def logistic_mixed(units: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    spline = SplineTransformer(n_knots=8, degree=3).fit(units[["doy"]].to_numpy()[train])
    onehot = OneHotEncoder(handle_unknown="ignore").fit(units[["province", "county"]].astype(str).to_numpy()[train])

    def x(mask):
        return np.hstack([spline.transform(units[["doy"]].to_numpy()[mask]),
                          onehot.transform(units[["province", "county"]].astype(str).to_numpy()[mask]).toarray()])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(x(train), units["positive"].to_numpy()[train])
    return m.predict_proba(x(test))[:, 1]


def tree_model(feat: pd.DataFrame, train: np.ndarray, test: np.ndarray, kind: str) -> np.ndarray:
    X = feat[FEATURES].to_numpy(float)
    y = feat["positive"].to_numpy()
    if kind == "random_forest":
        fill = np.nanmean(X[train], axis=0)
        X = np.where(np.isnan(X), fill, X)
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, random_state=ver.SEED, n_jobs=-1)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                           random_state=ver.SEED)
    m.fit(X[train], y[train])
    return m.predict_proba(X[test])[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    all_units = pd.read_csv(args.out / "results" / "verification_units.csv", dtype={"province": str})
    rows = []
    for model, units in all_units.groupby("model", sort=False):
        units = units.reset_index(drop=True)
        d = encode(units)
        feat = history_features(units)
        names = ["county_calendar", "county_unpooled", "persistence", "logistic_mixed", "random_forest",
                 "gradient_boosting"]
        preds = {k: np.full(len(units), np.nan) for k in names}
        for train, test in rolling_folds(d):
            preds["county_calendar"][test] = predict_on(fit_on(d, train), d, test, LEVELS["county_calendar"])
            preds["county_unpooled"][test] = predict_on(fit_on(d, train, prior=1e-6), d, test, LEVELS["county_calendar"])
            preds["persistence"][test] = persistence(feat, test)
            preds["logistic_mixed"][test] = logistic_mixed(units, train, test)
            for kind in ("random_forest", "gradient_boosting"):
                preds[kind][test] = tree_model(feat, train, test, kind)
        tested = ~np.isnan(preds["county_calendar"])
        sub = units[tested].reset_index(drop=True)
        obs = sub["positive"].to_numpy()
        p = {k: v[tested] for k, v in preds.items()}
        w = ver.bootstrap_weights(sub, "year", N_BOOT)
        draws = {k: ref.weighted_auc(w, v, obs) for k, v in p.items()}
        for k, v in p.items():
            row = {"model": model, "design": "rolling_origin", "forecast": k, "units": len(sub),
                   "positives": int(obs.sum()), "AUC": ref.auc(v, obs)}
            row["AUC_year_low"], row["AUC_year_high"] = ver.interval(draws[k])
            if k != "county_calendar":
                row["county_calendar_minus_this"] = ref.auc(p["county_calendar"], obs) - row["AUC"]
                lo, hi = ver.interval(draws["county_calendar"] - draws[k])
                row["county_calendar_minus_this_year_low"], row["county_calendar_minus_this_year_high"] = lo, hi
            if k != "persistence":
                row["Brier"] = brier(v, obs)
            # yearly AUC difference sign (bitter rot has two test years)
            yrs = sub["year"].to_numpy()
            diff = [ref.auc(p["county_calendar"][yrs == y], obs[yrs == y]) - ref.auc(v[yrs == y], obs[yrs == y])
                    for y in np.unique(yrs)]
            row["AUC_by_year"] = ";".join(f"{y}:{ref.auc(v[yrs == y], obs[yrs == y]):.4f}" for y in np.unique(yrs))
            row["county_calendar_minus_this_by_year"] = ";".join(f"{y}:{x:.4f}" for y, x in zip(np.unique(yrs), diff))
            row["test_years"] = len(diff)
            row["years_county_calendar_ahead"] = int(sum(x > 0 for x in diff))
            rows.append(row)
        print(model, "done", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "tables" / "M15_competing_models.csv", index=False)


if __name__ == "__main__":
    main()
