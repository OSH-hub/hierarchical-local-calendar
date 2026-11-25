"""Robustness of the local calendar: settings, county-label permutation, survey schedule, value, county table.

Usage
-----
python src/run_local_checks.py [--out <dir>]

Reads results/verification_units.csv and results/local_calendar_predictions.csv.
Writes tables/M04 ... M14.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import reference as ref
import verification as ver
from run_local_model import LEVELS, encode, fit_on, predict_on, rolling_folds
from run_reanalysis import ALPHAS, N_BOOT, THRESHOLDS

BANDWIDTHS = (5.0, 10.0, 15.0, 20.0)
PRIORS = (1.0, 2.0, 4.0, 8.0)
N_LABEL_PERM = 999
FORECASTS = ("national_calendar", "province_calendar", "county_calendar")


def rolling_predictions(d: dict, level: str = "county_calendar", predict_as: np.ndarray | None = None,
                        **settings) -> np.ndarray:
    """Rolling-origin predictions; ``predict_as`` replaces the county whose fitted calendar each unit receives."""
    target = d if predict_as is None else {**d, "county": predict_as}
    pred = np.full(len(d["doy"]), np.nan)
    for train, test in rolling_folds(d):
        pred[test] = predict_on(fit_on(d, train, **settings), target, test, LEVELS[level])
    return pred


def settings_rows(d: dict, model: str) -> list[dict]:
    rows = []
    for bw in BANDWIDTHS:
        for prior in PRIORS:
            row = {"model": model, "bandwidth_days": bw, "prior_events": prior}
            for level in ("national_calendar", "county_calendar"):
                p = rolling_predictions(d, level, bandwidth=bw, prior=prior)
                ok = ~np.isnan(p)
                row[f"AUC_{level}"] = ref.auc(p[ok], d["pos"][ok])
            row["AUC_gain"] = row["AUC_county_calendar"] - row["AUC_national_calendar"]
            rows.append(row)
    return rows


def label_permutation_row(d: dict, model: str, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Is the county level informative beyond the province level?

    Calendars are fitted with the true county labels.  In each permutation every
    test unit receives the fitted calendar of another county drawn from the same
    province (one shuffle of counties within provinces, applied to all years).
    The province part is unchanged, so the null isolates the county-specific part.
    """
    def county_auc(county_codes: np.ndarray) -> float:
        p = rolling_predictions(d, predict_as=county_codes)
        ok = ~np.isnan(p)
        return ref.auc(p[ok], d["pos"][ok])

    observed = county_auc(d["county"])
    null = np.empty(N_LABEL_PERM)
    for i in range(N_LABEL_PERM):
        mapping = np.arange(len(d["prov_of"]))
        for prov in range(d["n_prov"]):
            members = np.flatnonzero(d["prov_of"] == prov)
            mapping[members] = rng.permutation(members)
        null[i] = county_auc(mapping[d["county"]])
    return {"model": model, "AUC_county_calendar": observed, "null_mean": float(null.mean()),
            "null_p975": float(np.quantile(null, 0.975)), "p_one_sided": ver.permutation_p(observed, null),
            "permutations": N_LABEL_PERM}, null


def schedule_rows(units: pd.DataFrame, preds: pd.DataFrame, model: str) -> list[dict]:
    """Does the county gain survive among county-years whose first survey fell on the usual date?"""
    per = units.groupby(["county", "year"])["doy"].agg(first="min").reset_index()
    typical = per["first"].median()
    subsets = {"all_county_years": per[["county", "year"]],
               "first_survey_within_7_days_of_median": per.loc[(per["first"] - typical).abs() <= 7, ["county", "year"]]}
    rows = []
    for label, keys in subsets.items():
        sub = preds.merge(keys, on=["county", "year"])
        obs = sub["positive"].to_numpy()
        rows.append({"model": model, "subset": label, "county_years": len(keys), "units": len(sub),
                     "positives": int(obs.sum()), "median_first_survey_doy": float(typical),
                     "first_survey_doy_sd": float(per["first"].std()),
                     "AUC_operational_risk": ref.auc(sub["score"].to_numpy(), obs),
                     **{f"AUC_{k}": ref.auc(sub[k].to_numpy(), obs) for k in FORECASTS}})
    return rows


def value_rows(preds: pd.DataFrame, model: str) -> list[dict]:
    """Cost-loss value when action is taken if the predicted probability exceeds the cost-loss ratio."""
    obs = preds["positive"].to_numpy()
    base = obs.mean()
    w = ver.bootstrap_weights(preds, "year", N_BOOT)
    wbase = (w @ obs.astype(float)) / w.sum(1)

    def value_of(forecast_by_alpha):
        v = np.empty(len(ALPHAS))
        vb = np.empty((N_BOOT, len(ALPHAS)))
        for j, a in enumerate(ALPHAS):
            f = forecast_by_alpha(a)
            sc = ver.scores(ver.table_counts(f, obs))
            v[j] = ver.relative_value(sc["POD"], sc["POFD"], base, a)
            dr = ver.scores(ver.weighted_counts(w, f, obs))
            vb[:, j] = ver.relative_value(dr["POD"], dr["POFD"], wbase, a)
        return v, vb

    score = preds["score"].to_numpy()
    curves = {name: value_of(lambda a, p=preds[name].to_numpy(): p >= a) for name in FORECASTS}
    risk = [value_of(lambda a, t=t: score >= t) for t in THRESHOLDS[model]]
    curves["operational_best_threshold"] = (np.max([r[0] for r in risk], axis=0), np.max([r[1] for r in risk], axis=0))
    cal = [value_of(lambda a, c=c: preds[c].to_numpy()) for c in ver.CALENDARS]
    curves["fixed_calendar_best"] = (np.max([r[0] for r in cal], axis=0), np.max([r[1] for r in cal], axis=0))
    rows = []
    for name, (v, vb) in curves.items():
        lo, hi = np.nanquantile(vb, [0.025, 0.975], axis=0)
        dlo, dhi = np.nanquantile(curves["county_calendar"][1] - vb, [0.025, 0.975], axis=0)
        for j, a in enumerate(ALPHAS):
            rows.append({"model": model, "forecast": name, "cost_loss_ratio": a, "base_rate": float(base),
                         "value": float(v[j]), "value_year_low": float(lo[j]), "value_year_high": float(hi[j]),
                         "county_minus_this_low": float(dlo[j]), "county_minus_this_high": float(dhi[j])})
    return rows


def county_table(units: pd.DataFrame, d: dict, model: str) -> pd.DataFrame:
    """The fitted product: for every county, when the local calendar puts the risk above the overall rate."""
    fitted = fit_on(d, np.ones(len(units), bool))
    counties = sorted(units["county"].unique())
    base = d["pos"].mean()
    days = np.arange(1, 367)
    rows = []
    for i, c in enumerate(counties):
        prob = fitted.predict(days, np.full(366, i), np.full(366, d["prov_of"][i]))
        high = days[prob >= base]
        own = units[units["county"] == c]
        rows.append({"model": model, "county": c, "province": own["province"].iloc[0],
                     "years_of_history": int(own["year"].nunique()), "events": int(own["positive"].sum()),
                     "timing_shift_days": float(fitted.shift_county[i]), "height_ratio": float(fitted.ratio_county[i]),
                     "peak_day_of_year": int(days[prob.argmax()]), "peak_probability": float(prob.max()),
                     "elevated_from_day": int(high.min()) if len(high) else np.nan,
                     "elevated_to_day": int(high.max()) if len(high) else np.nan,
                     "elevated_days": int(len(high))})
    return pd.DataFrame(rows)


def yearly_rows(preds: pd.DataFrame, model: str) -> list[dict]:
    """ROC area of each forecast in each rolling-origin test year."""
    rows = []
    for year, g in preds.groupby("year"):
        obs = g["positive"].to_numpy()
        if obs.sum() < 3:
            continue
        rows.append({"model": model, "year": int(year), "units": len(g), "events": int(obs.sum()),
                     "AUC_operational_risk": ref.auc(g["score"].to_numpy(), obs),
                     **{f"AUC_{k}": ref.auc(g[k].to_numpy(), obs) for k in FORECASTS}})
    return rows


def reliability_rows(preds: pd.DataFrame, model: str, n_bins: int = 6) -> list[dict]:
    """Observed event frequency in equal-count bins of the predicted probability."""
    rows = []
    for name in ("national_calendar", "county_calendar"):
        g = preds[[name, "positive"]].sort_values(name).reset_index(drop=True)
        for b, idx in enumerate(np.array_split(np.arange(len(g)), n_bins)):
            part = g.iloc[idx]
            rows.append({"model": model, "forecast": name, "bin": b + 1, "units": len(part),
                         "mean_predicted": float(part[name].mean()), "observed_frequency": float(part["positive"].mean())})
    return rows


def updating_rows(d: dict, preds: pd.DataFrame, model: str, first_test_year: int = 2020) -> list[dict]:
    """Updated (rolling-origin) and frozen (fitted through the year before) calendars on the same test years."""
    train, test = d["year"] < first_test_year, d["year"] >= first_test_year
    if d["pos"][train].sum() < 10:
        return []
    frozen = fit_on(d, train)
    sub = preds[preds["year"] >= first_test_year].reset_index(drop=True)
    keys = pd.DataFrame({"county": d["county"][test], "year": d["year"][test], "doy": d["doy"][test],
                         "frozen_county": predict_on(frozen, d, test, LEVELS["county_calendar"]),
                         "frozen_national": predict_on(frozen, d, test, LEVELS["national_calendar"])})
    lookup = pd.Series(range(len(d["prov_of"])), index=sorted(preds["county"].unique()))
    sub = sub.assign(county_code=sub["county"].map(lookup))
    merged = sub.merge(keys, left_on=["county_code", "year", "doy"], right_on=["county", "year", "doy"],
                       suffixes=("", "_k"), validate="one_to_one")
    obs = merged["positive"].to_numpy()
    w = ver.bootstrap_weights(merged, "year", N_BOOT)
    cols = {"operational_risk": "score", "updated_national": "national_calendar", "updated_county": "county_calendar",
            "frozen_national": "frozen_national", "frozen_county": "frozen_county"}
    draws = {k: ref.weighted_auc(w, merged[c].to_numpy(), obs) for k, c in cols.items()}
    row = {"model": model, "test_years": f"{first_test_year}-{int(merged['year'].max())}", "units": len(merged),
           "events": int(obs.sum()), **{f"AUC_{k}": ref.auc(merged[c].to_numpy(), obs) for k, c in cols.items()}}
    for name, a, b in (("updated_county_minus_frozen_county", "updated_county", "frozen_county"),
                       ("updated_county_minus_updated_national", "updated_county", "updated_national"),
                       ("frozen_county_minus_frozen_national", "frozen_county", "frozen_national"),
                       ("updated_county_minus_operational", "updated_county", "operational_risk"),
                       ("frozen_county_minus_operational", "frozen_county", "operational_risk")):
        row[name] = row[f"AUC_{a}"] - row[f"AUC_{b}"]
        row[f"{name}_year_low"], row[f"{name}_year_high"] = ver.interval(draws[a] - draws[b])
    return [row]


def roc_rows(preds: pd.DataFrame, model: str) -> list[dict]:
    """Points of the ROC curve of each forecast, and of the fixed July-August calendar."""
    obs = preds["positive"].to_numpy()
    rows = []
    for name, col in (("operational_risk", "score"), *((k, k) for k in FORECASTS)):
        score = preds[col].to_numpy()
        cuts = np.r_[np.inf, np.unique(score)[::-1], -np.inf]
        for cut in cuts:
            rows.append({"model": model, "forecast": name, "threshold": cut,
                         "POFD": float((score[~obs] >= cut).mean()), "POD": float((score[obs] >= cut).mean())})
    cal = preds["cal_jul_aug"].to_numpy()
    rows.append({"model": model, "forecast": "cal_jul_aug", "threshold": np.nan,
                 "POFD": float(cal[~obs].mean()), "POD": float(cal[obs].mean())})
    return rows


def calendar_curves(units: pd.DataFrame, d: dict, model: str) -> pd.DataFrame:
    """Daily probability curves of the calendar fitted on all years: national, and every county."""
    fitted = fit_on(d, np.ones(len(units), bool))
    counties = sorted(units["county"].unique())
    days = np.arange(1, 367)
    frames = [pd.DataFrame({"model": model, "level": "national", "county": "", "day_of_year": days,
                            "probability": fitted.predict(days, np.zeros(366, int), np.zeros(366, int), "national")})]
    for i, c in enumerate(counties):
        frames.append(pd.DataFrame({"model": model, "level": "county", "county": c, "day_of_year": days,
                                    "probability": fitted.predict(days, np.full(366, i), np.full(366, d["prov_of"][i]))}))
    return pd.concat(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--only-permutation", action="store_true")
    args = ap.parse_args()
    all_units = pd.read_csv(args.out / "results" / "verification_units.csv", dtype={"province": str})
    all_preds = pd.read_csv(args.out / "results" / "local_calendar_predictions.csv", dtype={"province": str})
    rng = np.random.default_rng(ver.SEED)
    m04, m05, m06, m07, m08, m09, m10, m11, m12, m14, nulls = [], [], [], [], [], [], [], [], [], [], []
    for model, units in all_units.groupby("model", sort=False):
        units = units.reset_index(drop=True)
        preds = all_preds[all_preds["model"] == model].reset_index(drop=True)
        d = encode(units)
        if not args.only_permutation:
            m04 += settings_rows(d, model)
        row, null = label_permutation_row(d, model, rng)
        m05.append(row)
        nulls.append(pd.DataFrame({"model": model, "AUC_permuted_county_labels": null}))
        if not args.only_permutation:
            m06 += schedule_rows(units, preds, model)
            m07 += value_rows(preds, model)
            m08.append(county_table(units, d, model))
            m09.append(calendar_curves(units, d, model))
            m10 += yearly_rows(preds, model)
            m11 += reliability_rows(preds, model)
            m12 += roc_rows(preds, model)
            m14 += updating_rows(d, preds, model)
        print(model, "done", flush=True)
    t = args.out / "tables"
    pd.DataFrame(m05).to_csv(t / "M05_county_label_permutation.csv", index=False)
    pd.concat(nulls).round(6).to_csv(t / "M13_county_label_permutation_null.csv", index=False)
    if args.only_permutation:
        return
    pd.DataFrame(m04).to_csv(t / "M04_settings_sensitivity.csv", index=False)
    pd.DataFrame(m06).to_csv(t / "M06_survey_schedule_check.csv", index=False)
    pd.DataFrame(m07).to_csv(t / "M07_value_curves_local_calendar.csv", index=False)
    pd.concat(m08).to_csv(t / "M08_county_calendar_table.csv", index=False)
    pd.concat(m09).round({"probability": 5}).to_csv(t / "M09_calendar_curves.csv", index=False)
    pd.DataFrame(m10).to_csv(t / "M10_scores_by_year.csv", index=False)
    pd.DataFrame(m11).to_csv(t / "M11_reliability.csv", index=False)
    pd.DataFrame(m12).round({"POFD": 6, "POD": 6}).to_csv(t / "M12_roc_points.csv", index=False)
    pd.DataFrame(m14).to_csv(t / "M14_updated_versus_frozen.csv", index=False)


if __name__ == "__main__":
    main()
