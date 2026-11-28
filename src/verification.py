from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from loader import Record

SEED = 20261026
CALENDARS = {"cal_jul_aug": ((7, 1), (8, 31)), "cal_jun15_sep15": ((6, 15), (9, 15))}


def _doy(ordinal: int) -> int:
    return dt.date.fromordinal(int(ordinal)).timetuple().tm_yday


def risk_by_doy(records: dict, model: str) -> tuple[dict, list[int]]:
    grid, years = {}, set()
    for (m, c, y), r in records.items():
        if m != model:
            continue
        a = np.full(367, np.nan)
        a[[_doy(x) for x in r.days]] = r.risk
        grid[c, y] = a
        years.add(y)
    return grid, sorted(years)


def build_units(records: dict, model: str, window: int = 45) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    grid, years = risk_by_doy(records, model)
    rows, donor = [], []
    for (m, c, y), r in records.items():
        if m != model or not len(r.obs):
            continue
        seen_positive = False
        for k, (o, v) in enumerate(zip(r.obs, r.values)):
            d = _doy(o)
            lo = max(1, d - window)
            vals = []
            for yy in years:
                seg = grid[c, yy][lo:d + 1] if (c, yy) in grid else np.array([np.nan])
                vals.append(np.nan if np.isnan(seg).all() else np.nanmax(seg))
            own = grid[c, y][lo:d + 1]
            row = {"model": m, "county": c, "year": y, "province": r.province, "survey_index": k,
                   "survey_day": int(o), "doy": d, "value": float(v), "positive": bool(v > 0),
                   "at_risk": not seen_positive, "first_survey": k == 0,
                   "window_days": int((~np.isnan(own)).sum()),
                   "score": 0.0 if np.isnan(own).all() else float(np.nanmax(own))}
            for name, (first, last) in CALENDARS.items():
                a, b = dt.date(y, *first).toordinal(), dt.date(y, *last).toordinal()
                row[name] = bool(o - window <= b and o >= a)
            rows.append(row)
            donor.append(vals)
            seen_positive = seen_positive or v > 0
    return pd.DataFrame(rows), np.array(donor, dtype=float), years


def table_counts(forecast: np.ndarray, observed: np.ndarray) -> np.ndarray:
    f, o = np.asarray(forecast, bool), np.asarray(observed, bool)
    return np.array([(f & o).sum(), (f & ~o).sum(), (~f & o).sum(), (~f & ~o).sum()], dtype=float)


def scores(counts: np.ndarray) -> dict:
    a, b, c, d = np.moveaxis(np.asarray(counts, float), -1, 0)
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        pod, pofd = a / (a + c), b / (b + d)
        lh, lf, l1h, l1f = np.log(pod), np.log(pofd), np.log(1 - pod), np.log(1 - pofd)
        out = {"base_rate": (a + c) / n, "warning_rate": (a + b) / n, "POD": pod, "POFD": pofd,
               "FAR_ratio": b / (a + b), "CSI": a / (a + b + c), "bias": (a + b) / (a + c),
               "PSS": pod - pofd, "SEDI": (lf - lh - l1f + l1h) / (lf + lh + l1f + l1h),
               "odds_ratio": (a * d) / (b * c)}
    return out


def auc_from_levels(level_counts: np.ndarray) -> np.ndarray:
    neg, pos = level_counts[..., 0], level_counts[..., 1]
    below = np.cumsum(neg, axis=-1) - neg
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pos * (below + 0.5 * neg)).sum(-1) / (pos.sum(-1) * neg.sum(-1))


def level_table(score: np.ndarray, observed: np.ndarray, levels: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(levels, score)
    out = np.zeros((len(levels), 2))
    np.add.at(out, (idx, np.asarray(observed, int)), 1)
    return out


def relative_value(pod, pofd, base, alpha) -> np.ndarray:
    alpha = np.asarray(alpha, float)
    climate = np.minimum(alpha, base)
    forecast = pofd * alpha * (1 - base) + pod * base * alpha + (1 - pod) * base
    with np.errstate(divide="ignore", invalid="ignore"):
        return (climate - forecast) / (climate - base * alpha)


def cluster_draws(labels: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    uniq, idx = np.unique(labels, return_inverse=True)
    pick = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    mult = np.zeros((n_boot, len(uniq)))
    np.add.at(mult, (np.arange(n_boot)[:, None], pick), 1)
    return idx, mult


def bootstrap_weights(units: pd.DataFrame, scheme: str, n_boot: int = 5000, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if scheme in ("year", "county"):
        idx, mult = cluster_draws(units[scheme].to_numpy(), n_boot, rng)
        return mult[:, idx]
    iy, my = cluster_draws(units["year"].to_numpy(), n_boot, rng)
    ic, mc = cluster_draws(units["county"].to_numpy(), n_boot, rng)
    return my[:, iy] * mc[:, ic]


def weighted_counts(weights: np.ndarray, forecast: np.ndarray, observed: np.ndarray) -> np.ndarray:
    f, o = np.asarray(forecast, float), np.asarray(observed, float)
    cells = np.stack([f * o, f * (1 - o), (1 - f) * o, (1 - f) * (1 - o)], axis=1)
    return weights @ cells


def weighted_levels(weights: np.ndarray, score: np.ndarray, observed: np.ndarray, levels: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(levels, score)
    o = np.asarray(observed, int)
    out = np.zeros((weights.shape[0], len(levels), 2))
    for l in range(len(levels)):
        for k in (0, 1):
            out[:, l, k] = weights[:, (idx == l) & (o == k)].sum(1)
    return out


def interval(draws: np.ndarray) -> tuple[float, float]:
    v = draws[np.isfinite(draws)]
    if len(v) < 0.9 * len(draws):
        return (np.nan, np.nan)
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(lo), float(hi)


def year_permutations(n_eval: int, n_donor: int, eval_pos: np.ndarray, n_perm: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, n_eval), dtype=int)
    for i in range(n_perm):
        out[i] = rng.permutation(n_donor)[:n_eval]
    return out


def permutation_p(observed: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    return float((1 + (null >= observed - 1e-12).sum()) / (len(null) + 1))
