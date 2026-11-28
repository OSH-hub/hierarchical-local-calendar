from __future__ import annotations

import numpy as np
import pandas as pd

BIN_DAYS = 15


def loyo_climatology(units: pd.DataFrame) -> np.ndarray:
    bins = (units["doy"].to_numpy() // BIN_DAYS).astype(int)
    year, pos = units["year"].to_numpy(), units["positive"].to_numpy().astype(float)
    out = np.empty(len(units))
    for y in np.unique(year):
        test, train = year == y, year != y
        num = np.bincount(bins[train], weights=pos[train], minlength=bins.max() + 1)
        den = np.bincount(bins[train], minlength=bins.max() + 1)
        prob = np.where(den > 0, (num + 0.5) / (den + 1), pos[train].mean())
        out[test] = prob[bins[test]]
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, n_iter: int = 50) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-x @ beta))
        w = p * (1 - p)
        step = np.linalg.solve(x.T @ (x * w[:, None]) + 1e-8 * np.eye(x.shape[1]), x.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def loyo_combined(units: pd.DataFrame, climatology: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logit = np.log(climatology / (1 - climatology))
    x = np.column_stack([np.ones(len(units)), logit, score])
    y = units["positive"].to_numpy().astype(float)
    year = units["year"].to_numpy()
    pred, coefs = np.empty(len(units)), []
    for yy in np.unique(year):
        beta = fit_logistic(x[year != yy], y[year != yy])
        pred[year == yy] = 1 / (1 + np.exp(-x[year == yy] @ beta))
        coefs.append(beta[2])
    return pred, np.array(coefs)


def weighted_auc(weights: np.ndarray, score: np.ndarray, observed: np.ndarray, chunk: int = 250) -> np.ndarray:
    order = np.argsort(score, kind="stable")
    s, o = np.asarray(score)[order], np.asarray(observed, float)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) > 0) + 1]
    out = np.empty(weights.shape[0])
    for i in range(0, weights.shape[0], chunk):
        w = weights[i:i + chunk][:, order]
        pos = np.add.reduceat(w * o, starts, axis=1)
        neg = np.add.reduceat(w * (1 - o), starts, axis=1)
        below = np.cumsum(neg, axis=1) - neg
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i:i + chunk] = (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))
    return out


def auc(score: np.ndarray, observed: np.ndarray) -> float:
    return float(weighted_auc(np.ones((1, len(score))), score, observed)[0])
