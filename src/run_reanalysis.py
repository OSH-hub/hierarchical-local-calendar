"""Survey-level verification, seasonal reference, year-permutation control and value curves."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import reference as ref
import verification as ver
from loader import MODELS, load_records

THRESHOLDS = {"AMB": (1, 2, 3), "ABR": (1, 2, 3), "RB_IR": (1, 2, 4)}
ACTION = {"AMB": 3, "ABR": 3, "RB_IR": 2}
PRIMARY = (("AMB", 3), ("ABR", 3), ("RB_IR", 2), ("RB_IR", 4))
WINDOW = 45
WINDOWS = (15, 30, 45, 60)
N_BOOT = 5000
N_PERM = 9999
ALPHAS = np.round(np.arange(0.01, 1.0, 0.01), 2)
SCHEMES = ("year", "county", "two_way")
SCORE_NAMES = ("POD", "POFD", "PSS", "CSI", "FAR_ratio", "bias", "SEDI")


def forecasts(units: pd.DataFrame, model: str) -> dict[str, np.ndarray]:
    out = {f"risk>={t}": (units["score"] >= t).to_numpy() for t in THRESHOLDS[model]}
    out.update({name: units[name].to_numpy() for name in ver.CALENDARS})
    return out


def levels_of(model: str) -> np.ndarray:
    return np.array((0,) + THRESHOLDS[model], dtype=float)


def frame_of(units: pd.DataFrame, donor: np.ndarray, frame: str) -> tuple[pd.DataFrame, np.ndarray]:
    keep = units["at_risk"].to_numpy() if frame == "first_detection" else np.ones(len(units), bool)
    return units[keep].reset_index(drop=True), donor[keep]


def contingency_rows(units: pd.DataFrame, model: str, label: dict, with_intervals: bool) -> list[dict]:
    obs = units["positive"].to_numpy()
    weights = {s: ver.bootstrap_weights(units, s, N_BOOT) for s in SCHEMES} if with_intervals else {}
    rows = []
    for name, f in forecasts(units, model).items():
        counts = ver.table_counts(f, obs)
        sc = ver.scores(counts)
        row = {**label, "model": model, "forecast": name, "units": len(units), "hits": int(counts[0]),
               "false_alarms": int(counts[1]), "misses": int(counts[2]), "correct_negatives": int(counts[3]),
               **{k: float(sc[k]) for k in ("base_rate", "warning_rate") + SCORE_NAMES}}
        for scheme, w in weights.items():
            draws = ver.scores(ver.weighted_counts(w, f, obs))
            for k in ("POD", "POFD", "PSS", "CSI"):
                row[f"{k}_{scheme}_low"], row[f"{k}_{scheme}_high"] = ver.interval(draws[k])
        rows.append(row)
    return rows


def discrimination_row(units: pd.DataFrame, model: str, label: dict) -> dict:
    obs = units["positive"].to_numpy()
    score = units["score"].to_numpy()
    clim = ref.loyo_climatology(units)
    comb, coefs = ref.loyo_combined(units, clim, score)
    row = {**label, "model": model, "units": len(units), "positives": int(obs.sum()),
           "AUC_risk": ref.auc(score, obs), "AUC_climatology": ref.auc(clim, obs), "AUC_combined": ref.auc(comb, obs),
           "risk_coefficient_mean": float(coefs.mean()), "risk_coefficient_min": float(coefs.min()),
           "risk_coefficient_max": float(coefs.max())}
    row["AUC_risk_minus_climatology"] = row["AUC_risk"] - row["AUC_climatology"]
    row["AUC_combined_minus_climatology"] = row["AUC_combined"] - row["AUC_climatology"]
    for scheme in SCHEMES:
        w = ver.bootstrap_weights(units, scheme, N_BOOT)
        a = {"risk": ref.weighted_auc(w, score, obs), "climatology": ref.weighted_auc(w, clim, obs),
             "combined": ref.weighted_auc(w, comb, obs)}
        for k, v in a.items():
            row[f"AUC_{k}_{scheme}_low"], row[f"AUC_{k}_{scheme}_high"] = ver.interval(v)
        for k in ("risk", "combined"):
            lo, hi = ver.interval(a[k] - a["climatology"])
            row[f"AUC_{k}_minus_climatology_{scheme}_low"], row[f"AUC_{k}_minus_climatology_{scheme}_high"] = lo, hi
    return row


def paired_rows(units: pd.DataFrame, model: str, label: dict) -> list[dict]:
    obs = units["positive"].to_numpy()
    fc = forecasts(units, model)
    weights = {s: ver.bootstrap_weights(units, s, N_BOOT) for s in SCHEMES}
    rows = []
    for t in THRESHOLDS[model]:
        for cal in ver.CALENDARS:
            a, b = fc[f"risk>={t}"], fc[cal]
            sa, sb = ver.scores(ver.table_counts(a, obs)), ver.scores(ver.table_counts(b, obs))
            row = {**label, "model": model, "threshold": t, "calendar": cal,
                   "hits_risk_only": int((a & ~b & obs).sum()), "hits_calendar_only": int((~a & b & obs).sum()),
                   "false_risk_only": int((a & ~b & ~obs).sum()), "false_calendar_only": int((~a & b & ~obs).sum())}
            for k in ("POD", "POFD", "PSS"):
                row[f"delta_{k}"] = float(sa[k] - sb[k])
            for scheme, w in weights.items():
                da, db = ver.scores(ver.weighted_counts(w, a, obs)), ver.scores(ver.weighted_counts(w, b, obs))
                for k in ("POD", "POFD", "PSS"):
                    row[f"delta_{k}_{scheme}_low"], row[f"delta_{k}_{scheme}_high"] = ver.interval(da[k] - db[k])
            rows.append(row)
    return rows


def permutation_rows(units: pd.DataFrame, donor: np.ndarray, years: list[int], model: str, label: dict,
                     donor_years: list[int] | None = None, n_perm: int = N_PERM, keep_null: bool = False):
    donor_years = donor_years or years
    cols = np.array([years.index(y) for y in donor_years])
    eval_years = sorted(units["year"].unique())
    if not set(eval_years) <= set(donor_years):
        raise ValueError("evaluated years must be donor years")
    pos = units["year"].map({y: i for i, y in enumerate(eval_years)}).to_numpy()
    obs = units["positive"].to_numpy()
    lv = levels_of(model)
    d = np.nan_to_num(donor[:, cols], nan=0.0)
    maps = ver.year_permutations(len(eval_years), len(cols), pos, n_perm)
    rows_idx = np.arange(len(units))

    def stats(score):
        out = {"AUC": float(ver.auc_from_levels(ver.level_table(score, obs, lv)))}
        for t in THRESHOLDS[model]:
            sc = ver.scores(ver.table_counts(score >= t, obs))
            out[f"POD>={t}"], out[f"POFD>={t}"], out[f"PSS>={t}"] = float(sc["POD"]), float(sc["POFD"]), float(sc["PSS"])
        return out

    observed = stats(units["score"].to_numpy())
    null = pd.DataFrame([stats(d[rows_idx, m[pos]]) for m in maps])
    rows = []
    for k, v in observed.items():
        rows.append({**label, "model": model, "statistic": k, "observed": v, "null_mean": float(null[k].mean()),
                     "null_p025": float(null[k].quantile(0.025)), "null_p975": float(null[k].quantile(0.975)),
                     "p_one_sided": ver.permutation_p(v, null[k].to_numpy()), "permutations": n_perm,
                     "evaluated_years": len(eval_years), "donor_years": len(cols)})
    return (rows, null) if keep_null else rows


def value_rows(units: pd.DataFrame, model: str, label: dict) -> tuple[list[dict], list[dict]]:
    obs = units["positive"].to_numpy()
    base = obs.mean()
    w = ver.bootstrap_weights(units, "year", N_BOOT)
    wbase = (w @ obs.astype(float)) / w.sum(1)
    curves, summary, store = [], [], {}
    for name, f in forecasts(units, model).items():
        sc = ver.scores(ver.table_counts(f, obs))
        v = ver.relative_value(sc["POD"], sc["POFD"], base, ALPHAS)
        dr = ver.scores(ver.weighted_counts(w, f, obs))
        vb = ver.relative_value(dr["POD"][:, None], dr["POFD"][:, None], wbase[:, None], ALPHAS[None, :])
        store[name] = (v, vb)
        lo, hi = np.nanquantile(vb, [0.025, 0.975], axis=0)
        for a, x, l, h in zip(ALPHAS, v, lo, hi):
            curves.append({**label, "model": model, "forecast": name, "cost_loss_ratio": a, "value": float(x),
                           "value_year_low": float(l), "value_year_high": float(h)})
        positive = ALPHAS[v > 0]
        summary.append({**label, "model": model, "forecast": name, "base_rate": float(base),
                        "max_value": float(np.nanmax(v)), "cost_loss_at_max": float(ALPHAS[np.nanargmax(v)]),
                        "positive_from": float(positive.min()) if len(positive) else np.nan,
                        "positive_to": float(positive.max()) if len(positive) else np.nan})
    risk_names = [f"risk>={t}" for t in THRESHOLDS[model]]
    env = np.max([store[n][0] for n in risk_names], axis=0)
    cal = np.max([store[n][0] for n in ver.CALENDARS], axis=0)
    better = ALPHAS[(env > cal) & (env > 0)]
    envb = np.max([store[n][1] for n in risk_names], axis=0)
    calb = np.max([store[n][1] for n in ver.CALENDARS], axis=0)
    share = ((envb > calb) & (envb > 0)).mean(0)
    robust = ALPHAS[share >= 0.975]
    summary.append({**label, "model": model, "forecast": "risk_envelope_vs_best_calendar", "base_rate": float(base),
                    "max_value": float(np.nanmax(env)), "cost_loss_at_max": float(ALPHAS[np.nanargmax(env)]),
                    "positive_from": float(better.min()) if len(better) else np.nan,
                    "positive_to": float(better.max()) if len(better) else np.nan,
                    "robust_from": float(robust.min()) if len(robust) else np.nan,
                    "robust_to": float(robust.max()) if len(robust) else np.nan})
    return curves, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--rice-daily", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    start = time.time()
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (args.out / "results").mkdir(exist_ok=True)
    records, flow = load_records(args.project, args.rice_daily)

    r01, r02, r03, r04, r05c, r05s, nulls = [], [], [], [], [], [], {}
    unit_tables = []
    for model in MODELS:
        zero_years = sorted({y for (m, c, y), r in records.items() if m == model}
                            - {y for (m, c, y), r in records.items() if m == model and (r.risk > 0).any()})
        for window in WINDOWS:
            units, donor, years = ver.build_units(records, model, window)
            for frame in ("first_detection", "all_surveys"):
                q, d = frame_of(units, donor, frame)
                label = {"frame": frame, "window": window, "subset": "all_years"}
                primary = window == WINDOW and frame == "first_detection"
                r01 += contingency_rows(q, model, label, with_intervals=window == WINDOW)
                if window == WINDOW or frame == "first_detection":
                    r02.append(discrimination_row(q, model, label))
                if primary:
                    unit_tables.append(q)
                    r03 += paired_rows(q, model, label)
                    rows, null = permutation_rows(q, d, years, model, label, keep_null=True)
                    r04 += rows
                    nulls[model] = null
                    curves, summ = value_rows(q, model, label)
                    r05c += curves
                    r05s += summ
                    subsets = {"exclude_2023": q["year"] != 2023, "through_2019": q["year"] <= 2019,
                               "2020_2025": q["year"] >= 2020, "prior_negative_only": ~(q["first_survey"] & q["positive"])}
                    if zero_years:
                        subsets["years_with_risk_series"] = ~q["year"].isin(zero_years)
                        subsets["all_zero_years"] = q["year"].isin(zero_years)
                    for name, mask in subsets.items():
                        sub, dsub = q[mask.to_numpy()].reset_index(drop=True), d[mask.to_numpy()]
                        if sub["positive"].sum() < 5 or sub["year"].nunique() < 2:
                            continue
                        lab = {**label, "subset": name}
                        r01 += contingency_rows(sub, model, lab, with_intervals=True)
                        if name != "all_zero_years":
                            r02.append(discrimination_row(sub, model, lab))
                        if name in ("exclude_2023", "years_with_risk_series"):
                            dy = [y for y in years if y not in zero_years] if name == "years_with_risk_series" else \
                                 [y for y in years if y != 2023]
                            r04 += permutation_rows(sub, dsub, years, model, lab, donor_years=dy)
                elif frame == "first_detection" or window == WINDOW:
                    r04 += permutation_rows(q, d, years, model, label)
        print(model, "done", round(time.time() - start), "s", flush=True)

    r04 = pd.DataFrame(r04)
    key = r04["statistic"].str.startswith("PSS") & (r04["window"] == WINDOW) & (r04["frame"] == "first_detection") \
        & (r04["subset"] == "all_years")
    prim = r04[key & r04.apply(lambda r: (r["model"], int(r["statistic"].split(">=")[1]) if ">=" in r["statistic"] else -1)
                               in PRIMARY, axis=1)].sort_values("p_one_sided")
    holm, running = {}, 0.0
    for rank, (idx, p) in enumerate(prim["p_one_sided"].items()):
        running = max(running, min(1.0, (len(prim) - rank) * p))
        holm[idx] = running
    r04["p_holm_primary"] = pd.Series(holm)

    pd.DataFrame(r01).to_csv(tables / "R01_contingency_scores.csv", index=False)
    pd.DataFrame(r02).to_csv(tables / "R02_discrimination_seasonal_reference.csv", index=False)
    pd.DataFrame(r03).to_csv(tables / "R03_paired_calendar_differences.csv", index=False)
    r04.to_csv(tables / "R04_year_permutation_control.csv", index=False)
    pd.DataFrame(r05c).to_csv(tables / "R05a_value_curves.csv", index=False)
    pd.DataFrame(r05s).to_csv(tables / "R05b_value_summary.csv", index=False)
    pd.concat([n.assign(model=m) for m, n in nulls.items()]).round(6).to_csv(tables / "R10_year_permutation_null.csv", index=False)
    pd.concat(unit_tables).drop(columns=["value"]).to_csv(args.out / "results" / "verification_units.csv", index=False)

    meta = {"flow": flow, "window_primary": WINDOW, "windows": WINDOWS, "bootstrap_draws": N_BOOT, "permutations": N_PERM,
            "seed": ver.SEED, "climatology_bin_days": ref.BIN_DAYS, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - start, 1)}
    (args.out / "results" / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                           encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
