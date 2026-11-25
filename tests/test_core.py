"""Unit tests for the scores, the resampling tools and the local calendar.  Run with: pytest -q"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import local_calendar as lc  # noqa: E402
import reference as ref  # noqa: E402
import verification as ver  # noqa: E402


def test_scores_known_table():
    s = ver.scores(np.array([30.0, 10.0, 20.0, 40.0]))
    assert s["POD"] == pytest.approx(0.6)
    assert s["POFD"] == pytest.approx(0.2)
    assert s["PSS"] == pytest.approx(0.4)
    assert s["CSI"] == pytest.approx(0.5)
    assert s["FAR_ratio"] == pytest.approx(0.25)


def test_auc_perfect_reversed_and_ties():
    obs = np.array([0, 0, 1, 1], bool)
    assert ref.auc(np.array([0.1, 0.2, 0.8, 0.9]), obs) == pytest.approx(1.0)
    assert ref.auc(np.array([0.9, 0.8, 0.2, 0.1]), obs) == pytest.approx(0.0)
    assert ref.auc(np.zeros(4), obs) == pytest.approx(0.5)


def test_level_auc_matches_continuous_auc():
    rng = np.random.default_rng(1)
    score = rng.integers(0, 4, 500).astype(float)
    obs = rng.random(500) < 0.2 + 0.1 * score
    levels = np.arange(4.0)
    assert ver.auc_from_levels(ver.level_table(score, obs, levels)) == pytest.approx(ref.auc(score, obs))


def test_permutation_p_never_zero():
    assert ver.permutation_p(10.0, np.zeros(99)) == pytest.approx(0.01)
    assert ver.permutation_p(0.0, np.ones(99)) == pytest.approx(1.0)


def test_relative_value_of_perfect_forecast_is_one():
    alpha = np.array([0.1, 0.3, 0.5])
    assert np.allclose(ver.relative_value(1.0, 0.0, 0.3, alpha), 1.0)
    assert ver.relative_value(1.0, 1.0, 0.3, alpha)[0] == pytest.approx(0.0)   # always warn equals climatology


def test_bootstrap_weights_keep_cluster_structure():
    units = pd.DataFrame({"year": [1, 1, 2, 2, 3], "county": list("aabbc")})
    w = ver.bootstrap_weights(units, "year", n_boot=50, seed=3)
    assert w.shape == (50, 5)
    assert np.all(w[:, 0] == w[:, 1]) and np.all(w[:, 2] == w[:, 3])


def _toy(n_county=6, n_year=8, seed=0):
    rng = np.random.default_rng(seed)
    ratio = np.linspace(0.3, 2.0, n_county)
    rows = []
    for c in range(n_county):
        for y in range(n_year):
            for doy in range(150, 260, 15):
                p = min(0.9, 0.25 * ratio[c] * np.exp(-0.5 * ((doy - 205) / 25) ** 2))
                hit = rng.random() < p
                rows.append((c, y, doy, hit))
                if hit:
                    break
    c, y, doy, hit = map(np.array, zip(*rows))
    return c, y, doy, hit.astype(bool), ratio


def test_local_calendar_recovers_county_order_and_shrinks():
    c, _, doy, hit, ratio = _toy(seed=2)
    prov_of = np.zeros(6, dtype=int)
    fitted = lc.fit(doy, hit, c, prov_of, 1)
    assert np.corrcoef(fitted.ratio_county, ratio)[0, 1] > 0.7
    strong = lc.fit(doy, hit, c, prov_of, 1, prior=1e6)
    assert np.allclose(strong.ratio_county, strong.ratio_province[0], atol=1e-3)
    p = fitted.predict(np.array([205]), np.array([5]), np.array([0]))
    assert 0 < p[0] < 1


def test_unseen_county_gets_province_calendar():
    c, _, doy, hit, _ = _toy(seed=4)
    keep = c != 5
    fitted = lc.fit(doy[keep], hit[keep], c[keep], np.zeros(6, dtype=int), 1)
    assert fitted.ratio_county[5] == pytest.approx(fitted.ratio_province[0])
    assert fitted.shift_county[5] == pytest.approx(fitted.shift_province[0])


def test_year_permutation_does_not_reject_uninformative_scores():
    rng = np.random.default_rng(5)
    n_year, n_unit = 10, 400
    year = rng.integers(0, n_year, n_unit)
    obs = rng.random(n_unit) < 0.2
    donor = rng.integers(0, 4, (n_unit, n_year)).astype(float)
    maps = ver.year_permutations(n_year, n_year, year, 499, seed=1)

    def stat(score):
        return float(ver.scores(ver.table_counts(score >= 2, obs))["PSS"])

    observed = stat(donor[np.arange(n_unit), year])
    null = np.array([stat(donor[np.arange(n_unit), m[year]]) for m in maps])
    assert ver.permutation_p(observed, null) > 0.01


def test_example_runs_end_to_end(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic_data.py"), str(tmp_path)], check=True)
    import run_example
    out = run_example.main(tmp_path)
    assert out["AMB"]["events"] > 20
    assert out["AMB"]["AUC_county_calendar"] > out["AMB"]["AUC_operational"]
