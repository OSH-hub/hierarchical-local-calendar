"""Out-of-sample seasonal gating, validation of the timing controls, and the data-flow table.

Usage
-----
python src/run_controls.py --project <project root> [--rice-daily <file>] [--out <dir>]

Writes tables/R06 ... R09 as CSV.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import verification as ver
from loader import MODELS, load_records
from run_reanalysis import ACTION, N_BOOT, N_PERM, THRESHOLDS, WINDOW, frame_of

GATE_STEP = 5            # day-of-year grid for fitted season gates
N_SIM = 200              # synthetic archives per model in the control validation
N_SIM_DRAWS = 199        # permutations or shifts per synthetic archive


# ---------------------------------------------------------------- R06 season gate fitted out of sample
def pss(forecast: np.ndarray, obs: np.ndarray) -> float:
    return float(ver.scores(ver.table_counts(forecast, obs))["PSS"])


def fit_gate(doy: np.ndarray, obs: np.ndarray, score: np.ndarray | None, thresholds) -> tuple:
    """Choose (threshold, first day, last day) maximising PSS; threshold None means the gate alone.

    Ties keep the earliest candidate in grid order, so the fit is deterministic.
    """
    grid = range(int(doy.min()) // GATE_STEP * GATE_STEP, int(doy.max()) + GATE_STEP, GATE_STEP)
    best = (-np.inf, None)
    for t in (thresholds if score is not None else (None,)):
        active = np.ones(len(doy), bool) if t is None else score >= t
        for a, b in itertools.combinations(grid, 2):
            value = pss(active & (doy >= a) & (doy <= b), obs)
            if value > best[0] + 1e-12:
                best = (value, (t, a, b))
    return best[1]


def apply_gate(rule: tuple, doy: np.ndarray, score: np.ndarray) -> np.ndarray:
    t, a, b = rule
    active = np.ones(len(doy), bool) if t is None else score >= t
    return active & (doy >= a) & (doy <= b)


def gate_rows(units: pd.DataFrame, model: str) -> list[dict]:
    doy, obs, score, year = (units[k].to_numpy() for k in ("doy", "positive", "score", "year"))
    designs = {"leave_one_year_out": [(year != y, year == y) for y in np.unique(year)]}
    if (obs & (year <= 2019)).sum() >= 10:
        designs["fit_through_2019_test_2020_2025"] = [(year <= 2019, year >= 2020)]
    rows = []
    for design, folds in designs.items():
        test_all = np.zeros(len(units), bool)
        pred = {"season_gate": np.zeros(len(units), bool), "risk_within_gate": np.zeros(len(units), bool)}
        rules = {k: [] for k in pred}
        for train, test in folds:
            test_all |= test
            for name, sc in (("season_gate", None), ("risk_within_gate", score)):
                rule = fit_gate(doy[train], obs[train], None if sc is None else sc[train], THRESHOLDS[model])
                pred[name][test] = apply_gate(rule, doy[test], score[test])
                rules[name].append(rule)
        sub = units[test_all].reset_index(drop=True)
        o = obs[test_all]
        fc = {**{k: v[test_all] for k, v in pred.items()}, f"risk>={ACTION[model]}": score[test_all] >= ACTION[model],
              **{c: units[c].to_numpy()[test_all] for c in ver.CALENDARS}}
        w = ver.bootstrap_weights(sub, "year", N_BOOT)
        draws = {k: ver.scores(ver.weighted_counts(w, f, o)) for k, f in fc.items()}
        for name, f in fc.items():
            sc = ver.scores(ver.table_counts(f, o))
            row = {"model": model, "design": design, "forecast": name, "test_units": int(test_all.sum()),
                   "test_positives": int(o.sum()), "POD": float(sc["POD"]), "POFD": float(sc["POFD"]),
                   "PSS": float(sc["PSS"]), "CSI": float(sc["CSI"]),
                   "fitted_rules": "; ".join(sorted({str(r) for r in rules.get(name, [])}))}
            row["PSS_year_low"], row["PSS_year_high"] = ver.interval(draws[name]["PSS"])
            for other in ("season_gate", f"risk>={ACTION[model]}"):
                if other != name:
                    lo, hi = ver.interval(draws[name]["PSS"] - draws[other]["PSS"])
                    tag = "gate" if other == "season_gate" else "risk"
                    row[f"PSS_minus_{tag}"] = float(sc["PSS"] - ver.scores(ver.table_counts(fc[other], o))["PSS"])
                    row[f"PSS_minus_{tag}_year_low"], row[f"PSS_minus_{tag}_year_high"] = lo, hi
            rows.append(row)
    return rows


# ---------------------------------------------------------------- R07/R08 timing controls
class Archive:
    """Daily risk as an array [county, year, day of year] with fast window maxima."""

    def __init__(self, records: dict, model: str):
        grid, self.years = ver.risk_by_doy(records, model)
        self.counties = sorted({c for c, _ in grid})
        self.daily = np.zeros((len(self.counties), len(self.years), 367))
        for (c, y), a in grid.items():
            self.daily[self.counties.index(c), self.years.index(y)] = np.nan_to_num(a, nan=0.0)

    def window_max(self, daily: np.ndarray, window: int) -> np.ndarray:
        padded = np.concatenate([np.zeros(daily.shape[:-1] + (window,)), daily], axis=-1)
        return np.lib.stride_tricks.sliding_window_view(padded, window + 1, axis=-1).max(-1)


def circular_shift_p(daily_cy: list[np.ndarray], eligible: list[tuple[int, int]], threshold: float,
                     n_draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Circular-shift control of the submitted analysis, on event county-years.

    daily_cy[i] is the risk series on the support of event i and eligible[i] the
    index range of its detection window.  Returns observed POD, null mean and p.
    """
    hits = np.zeros(n_draws)
    actual = 0
    for series, (a, b) in zip(daily_cy, eligible):
        warn = series >= threshold
        n = len(warn)
        csum = np.concatenate([[0], np.cumsum(np.tile(warn, 2))])
        starts = (a - np.arange(n)) % n                       # window start after shifting by k days
        any_hit = (csum[starts + (b - a + 1)] - csum[starts]) > 0
        actual += any_hit[0]
        hits += any_hit[rng.integers(0, n, n_draws)]
    pod, null = actual / len(daily_cy), hits / len(daily_cy)
    return float(pod), float(null.mean()), ver.permutation_p(pod, null)


def control_rows(records: dict, model: str, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    arch = Archive(records, model)
    units, donor, years = ver.build_units(records, model, WINDOW)
    q, _ = frame_of(units, donor, "first_detection")
    ci = q["county"].map({c: i for i, c in enumerate(arch.counties)}).to_numpy()
    yi = q["year"].map({y: i for i, y in enumerate(arch.years)}).to_numpy()
    doy, obs = q["doy"].to_numpy(), q["positive"].to_numpy()
    eval_cols = np.array(sorted(set(yi)))
    pos = np.searchsorted(eval_cols, yi)
    t = ACTION[model]

    # supports of event county-years for the circular shift: first survey - W to last survey
    span = units.groupby(["county", "year"])["doy"].agg(["min", "max"])
    events = q[q["positive"]]
    ev = [(arch.counties.index(r.county), arch.years.index(r.year), max(1, span.loc[(r.county, r.year), "min"] - WINDOW),
           span.loc[(r.county, r.year), "max"], r.doy) for r in events.itertuples()]

    def both_controls(daily: np.ndarray, n_draws: int) -> dict:
        wmax = arch.window_max(daily, WINDOW)
        score = wmax[ci, yi, doy]
        don = wmax[ci[:, None], np.arange(len(arch.years))[None, :], doy[:, None]]
        maps = ver.year_permutations(len(eval_cols), len(arch.years), pos, n_draws, seed=int(rng.integers(1 << 31)))
        obs_sc = ver.scores(ver.table_counts(score >= t, obs))
        null = np.array([[s["POD"], s["PSS"]] for s in
                         (ver.scores(ver.table_counts(don[np.arange(len(q)), m[pos]] >= t, obs)) for m in maps)])
        series = [daily[c, y, lo:hi + 1] for c, y, lo, hi, _ in ev]
        elig = [(max(d - WINDOW, lo) - lo, d - lo) for _, _, lo, _, d in ev]
        pod, shift_mean, shift_p = circular_shift_p(series, elig, t, n_draws, rng)
        return {"POD": float(obs_sc["POD"]), "PSS": float(obs_sc["PSS"]),
                "year_perm_POD_mean": float(null[:, 0].mean()), "year_perm_PSS_mean": float(null[:, 1].mean()),
                "p_year_perm_POD": ver.permutation_p(obs_sc["POD"], null[:, 0]),
                "p_year_perm_PSS": ver.permutation_p(obs_sc["PSS"], null[:, 1]),
                "circular_POD": pod, "circular_POD_mean": shift_mean, "p_circular_POD": shift_p}

    real = {"model": model, "threshold": t, "archive": "observed", "draws": N_PERM, **both_controls(arch.daily, N_PERM)}
    sims = []
    nc, ny = arch.daily.shape[:2]
    for _ in range(N_SIM):
        # a warning archive with the real seasonal pattern and no link to the outcome:
        # every county-year receives the series of a random county in a different random year
        src_c = rng.integers(0, nc, (nc, ny))
        src_y = (np.arange(ny)[None, :] + rng.integers(1, ny, (nc, ny))) % ny
        sims.append(both_controls(arch.daily[src_c, src_y], N_SIM_DRAWS))
    sims = pd.DataFrame(sims)
    summary = {"model": model, "threshold": t, "synthetic_archives": N_SIM, "draws_each": N_SIM_DRAWS,
               **{f"rejection_rate_{k[2:]}": float((sims[k] <= 0.05).mean())
                  for k in ("p_circular_POD", "p_year_perm_POD", "p_year_perm_PSS")},
               "synthetic_POD_mean": float(sims["POD"].mean()), "synthetic_PSS_mean": float(sims["PSS"].mean())}
    return [real], [summary]


# ---------------------------------------------------------------- R09 data flow
def flow_rows(records: dict, flow: dict) -> list[dict]:
    rows = []
    for model in MODELS:
        recs = [r for (m, _, _), r in records.items() if m == model]
        sup = [r for r in recs if len(r.obs)]
        units, donor, _ = ver.build_units(records, model, WINDOW)
        first = units[units["at_risk"]]
        rows.append({
            "model": model,
            "rows_through_2025": flow["item_rows_through_2025"][model],
            "rows_with_numeric_value": flow["numeric_rows"][model],
            "rows_in_counties_with_risk_series": flow["rows_in_counties_with_risk"][model],
            "counties_surveyed": flow["survey_counties_total"][model],
            "counties_with_risk_series": flow["risk_counties"][model],
            "risk_county_years": len(recs),
            "county_years_with_valid_survey": len(sup),
            "counties_with_valid_survey": len({r.county for r in sup}),
            "years_with_valid_survey": len({r.year for r in sup}),
            "valid_surveys": len(units),
            "positive_surveys": int(units["positive"].sum()),
            "first_detection_units": len(first),
            "event_county_years": int(first["positive"].sum()),
            "events_at_first_survey": int((first["positive"] & first["first_survey"]).sum()),
            "negative_county_years": len(sup) - int(first["positive"].sum()),
            "event_counties": int(first.loc[first["positive"], "county"].nunique()),
            "units_with_truncated_window": int((first["window_days"] < WINDOW + 1).sum()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--rice-daily", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records, flow = load_records(args.project, args.rice_daily)
    rng = np.random.default_rng(ver.SEED)

    pd.DataFrame(flow_rows(records, flow)).to_csv(tables / "R09_data_flow.csv", index=False)
    r06, r07, r08 = [], [], []
    for model in MODELS:
        units, donor, _ = ver.build_units(records, model, WINDOW)
        q, _ = frame_of(units, donor, "first_detection")
        r06 += gate_rows(q, model)
        real, sim = control_rows(records, model, rng)
        r07 += real
        r08 += sim
        print(model, "done", flush=True)
    pd.DataFrame(r06).to_csv(tables / "R06_season_gate_out_of_sample.csv", index=False)
    pd.DataFrame(r07).to_csv(tables / "R07_timing_controls_observed.csv", index=False)
    pd.DataFrame(r08).to_csv(tables / "R08_timing_controls_validation.csv", index=False)


if __name__ == "__main__":
    main()
