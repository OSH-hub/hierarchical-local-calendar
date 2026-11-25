"""Evaluate the hierarchical local calendar against the national references.

Reads results/verification_units.csv written by run_reanalysis.py.

Usage
-----
python src/run_local_model.py [--out <dir>]

Writes
  tables/M01_local_calendar_scores.csv     leave-one-year-out and temporal-split scores, all comparators
  tables/M02_learning_curve.csv            skill against the number of training years
  tables/M03_new_county_adaptation.csv     skill for a county unseen in training, then with k of its own years
  results/local_calendar_predictions.csv   out-of-year predictions per unit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import local_calendar as lc
import reference as ref
import verification as ver
from run_reanalysis import ACTION, N_BOOT

LEVELS = {"national_calendar": ("national", True, True), "province_calendar": ("province", True, True),
          "county_calendar": ("county", True, True), "county_timing_only": ("county", True, False),
          "county_height_only": ("county", False, True)}
N_REPEAT = 20           # random draws of training years (learning curve) and own years (adaptation)
MAX_OWN_YEARS = 5
MIN_HISTORY = 3         # years of history before the first rolling-origin test year


def encode(units: pd.DataFrame) -> dict:
    counties = sorted(units["county"].unique())
    provinces = sorted(units["province"].astype(str).unique())
    c = units["county"].map({k: i for i, k in enumerate(counties)}).to_numpy()
    p = units["province"].astype(str).map({k: i for i, k in enumerate(provinces)}).to_numpy()
    prov_of = np.zeros(len(counties), dtype=int)
    prov_of[c] = p
    return {"doy": units["doy"].to_numpy(), "pos": units["positive"].to_numpy(), "county": c, "province": p,
            "year": units["year"].to_numpy(), "prov_of": prov_of, "n_prov": len(provinces)}


def fit_on(d: dict, mask: np.ndarray, **settings) -> lc.LocalCalendar:
    return lc.fit(d["doy"][mask], d["pos"][mask], d["county"][mask], d["prov_of"], d["n_prov"], **settings)


def rolling_folds(d: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each test year is predicted from all earlier years; at least MIN_HISTORY years of history."""
    years = np.unique(d["year"])
    return [(d["year"] < y, d["year"] == y) for y in years[MIN_HISTORY:]]


def predict_on(model: lc.LocalCalendar, d: dict, mask: np.ndarray, spec: tuple) -> np.ndarray:
    level, shift, ratio = spec
    return model.predict(d["doy"][mask], d["county"][mask], d["province"][mask], level, shift, ratio)


def brier(p: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((p - o) ** 2))


def matched_scores(score: np.ndarray, obs: np.ndarray, rate: float) -> dict:
    """POD, POFD and PSS when the highest-scoring units are warned, as many as the reference warns."""
    k = int(round(rate * len(score)))
    order = np.argsort(-score, kind="stable")
    warn = np.zeros(len(score), bool)
    warn[order[:k]] = True
    sc = ver.scores(ver.table_counts(warn, obs))
    return {"POD": float(sc["POD"]), "POFD": float(sc["POFD"]), "PSS": float(sc["PSS"])}


def score_rows(units: pd.DataFrame, preds: dict, model: str, design: str) -> list[dict]:
    obs = units["positive"].to_numpy()
    w = ver.bootstrap_weights(units, "year", N_BOOT)
    action = (units["score"] >= ACTION[model]).to_numpy()
    cal = units["cal_jul_aug"].to_numpy()
    auc_draws = {k: ref.weighted_auc(w, p, obs) for k, p in preds.items()}
    rows = []
    for name, p in preds.items():
        row = {"model": model, "design": design, "forecast": name, "units": len(units), "positives": int(obs.sum()),
               "AUC": ref.auc(p, obs)}
        row["AUC_year_low"], row["AUC_year_high"] = ver.interval(auc_draws[name])
        for other in ("operational_risk", "national_calendar"):
            if other != name:
                row[f"AUC_minus_{other}"] = row["AUC"] - ref.auc(preds[other], obs)
                lo, hi = ver.interval(auc_draws[name] - auc_draws[other])
                row[f"AUC_minus_{other}_year_low"], row[f"AUC_minus_{other}_year_high"] = lo, hi
        if name != "operational_risk":
            row["Brier"] = brier(p, obs)
            row["Brier_skill_vs_national"] = 1 - brier(p, obs) / brier(preds["national_calendar"], obs)
        for tag, f in (("operational", action), ("jul_aug", cal)):
            m = matched_scores(p, obs, f.mean())
            row.update({f"{k}_at_{tag}_warning_rate": v for k, v in m.items()})
        rows.append(row)
    for tag, f in (("operational_warning", action), ("calendar_jul_aug", cal)):
        sc = ver.scores(ver.table_counts(f, obs))
        rows.append({"model": model, "design": design, "forecast": f"reference_{tag}", "units": len(units),
                     "positives": int(obs.sum()), "warning_rate": float(f.mean()), "POD": float(sc["POD"]),
                     "POFD": float(sc["POFD"]), "PSS": float(sc["PSS"])})
    return rows


def out_of_year(d: dict, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    preds = {k: np.full(len(d["doy"]), np.nan) for k in LEVELS}
    for train, test in folds:
        model = fit_on(d, train)
        for name, spec in LEVELS.items():
            preds[name][test] = predict_on(model, d, test, spec)
    return preds


def learning_curve(d: dict, model: str, rng: np.random.Generator) -> list[dict]:
    years = np.unique(d["year"])
    rows = []
    for n in range(1, len(years)):
        aucs = {k: [] for k in ("national_calendar", "county_calendar")}
        for _ in range(N_REPEAT if n < len(years) - 1 else 1):
            preds = {k: np.full(len(d["doy"]), np.nan) for k in aucs}
            for y in years:
                others = years[years != y]
                chosen = rng.choice(others, n, replace=False)
                train = np.isin(d["year"], chosen)
                if d["pos"][train].sum() < 2:
                    continue
                fitted = fit_on(d, train)
                for k in aucs:
                    preds[k][d["year"] == y] = predict_on(fitted, d, d["year"] == y, LEVELS[k])
            ok = ~np.isnan(preds["county_calendar"])
            for k in aucs:
                aucs[k].append(ref.auc(preds[k][ok], d["pos"][ok]))
        rows.append({"model": model, "training_years": n,
                     **{f"AUC_{k}_mean": float(np.mean(v)) for k, v in aucs.items()},
                     **{f"AUC_{k}_min": float(np.min(v)) for k, v in aucs.items()},
                     **{f"AUC_{k}_max": float(np.max(v)) for k, v in aucs.items()}})
    return rows


def new_county_adaptation(d: dict, model: str, rng: np.random.Generator) -> list[dict]:
    """A county absent from training receives the province calendar; then k of its own years are added."""
    years = np.unique(d["year"])
    n_county = len(d["prov_of"])
    rows = []
    for k in range(MAX_OWN_YEARS + 1):
        aucs, briers = [], []
        for _ in range(N_REPEAT if k else 1):
            pred = np.full(len(d["doy"]), np.nan)
            for y in years:
                base = d["year"] != y
                for c in range(n_county):
                    test = (d["county"] == c) & (d["year"] == y)
                    if not test.any():
                        continue
                    own_years = np.unique(d["year"][(d["county"] == c) & base])
                    if len(own_years) < k:
                        continue
                    chosen = rng.choice(own_years, k, replace=False) if k else np.array([], dtype=int)
                    train = base & ((d["county"] != c) | np.isin(d["year"], chosen))
                    fitted = fit_on(d, train)
                    pred[test] = predict_on(fitted, d, test, LEVELS["county_calendar"])
            ok = ~np.isnan(pred)
            if d["pos"][ok].sum() < 5:      # too few counties have k other years of their own
                break
            aucs.append(ref.auc(pred[ok], d["pos"][ok]))
            briers.append(brier(pred[ok], d["pos"][ok]))
        if not aucs:
            continue
        rows.append({"model": model, "own_years_added": k, "units": int(ok.sum()), "AUC_mean": float(np.mean(aucs)),
                     "AUC_min": float(np.min(aucs)), "AUC_max": float(np.max(aucs)), "Brier_mean": float(np.mean(briers))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--skip-adaptation", action="store_true")
    args = ap.parse_args()
    all_units = pd.read_csv(args.out / "results" / "verification_units.csv", dtype={"province": str})
    rng = np.random.default_rng(ver.SEED)
    m01, m02, m03, keep = [], [], [], []
    for model, units in all_units.groupby("model", sort=False):
        units = units.reset_index(drop=True)
        d = encode(units)
        years = np.unique(d["year"])
        designs = {"rolling_origin": rolling_folds(d), "leave_one_year_out": [(d["year"] != y, d["year"] == y) for y in years]}
        if (d["pos"] & (d["year"] <= 2019)).sum() >= 10:
            designs["fit_through_2019_test_2020_2025"] = [(d["year"] <= 2019, d["year"] >= 2020)]
        for design, folds in designs.items():
            preds = out_of_year(d, folds)
            tested = ~np.isnan(preds["county_calendar"])
            sub = units[tested].reset_index(drop=True)
            p = {"operational_risk": sub["score"].to_numpy(), **{k: v[tested] for k, v in preds.items()}}
            m01 += score_rows(sub, p, model, design)
            if design == "rolling_origin":
                keep.append(sub[["model", "county", "province", "year", "doy", "positive", "score", "cal_jul_aug",
                                 "cal_jun15_sep15"]].assign(
                    **{k: v[tested] for k, v in preds.items()}))
        m02 += learning_curve(d, model, rng)
        if not args.skip_adaptation:
            m03 += new_county_adaptation(d, model, rng)
        print(model, "done", flush=True)
    pd.DataFrame(m01).to_csv(args.out / "tables" / "M01_local_calendar_scores.csv", index=False)
    pd.DataFrame(m02).to_csv(args.out / "tables" / "M02_learning_curve.csv", index=False)
    if m03:
        pd.DataFrame(m03).to_csv(args.out / "tables" / "M03_new_county_adaptation.csv", index=False)
    pd.concat(keep).to_csv(args.out / "results" / "local_calendar_predictions.csv", index=False)


if __name__ == "__main__":
    main()
