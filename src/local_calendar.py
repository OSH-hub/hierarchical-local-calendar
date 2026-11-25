"""Hierarchical local calendar: first-detection hazard by day of year, adapted to province and county.

The model has three nested levels.

national   h(d): kernel-smoothed fraction of at-risk surveys on day of year d that turned positive
province   the national curve shifted in time and scaled in height for the province
county     the province curve shifted and scaled again for the county

Shift and scale of a unit are shrunk towards its parent level, so a county with
little history stays close to its province and a county with a long history
follows its own record.  No weather or archived warning enters the model.

Fixed settings (chosen before evaluation): Gaussian kernel bandwidth 10 days,
prior strength 2 events for both the height ratio and the timing shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BANDWIDTH = 10.0
PRIOR_EVENTS = 2.0
_DAYS = np.arange(367)
_KERNELS: dict[float, np.ndarray] = {}


def _kernel(bandwidth: float) -> np.ndarray:
    if bandwidth not in _KERNELS:
        _KERNELS[bandwidth] = np.exp(-0.5 * ((_DAYS[:, None] - _DAYS[None, :]) / bandwidth) ** 2)
    return _KERNELS[bandwidth]


@dataclass
class LocalCalendar:
    hazard: np.ndarray           # national hazard by day of year, length 367
    shift_province: np.ndarray   # days, per province
    ratio_province: np.ndarray
    shift_county: np.ndarray     # days, per county (already includes the province shift)
    ratio_county: np.ndarray

    def predict(self, doy: np.ndarray, county: np.ndarray, province: np.ndarray, level: str = "county",
                use_shift: bool = True, use_ratio: bool = True) -> np.ndarray:
        """Probability that an at-risk survey on ``doy`` is positive, at the requested level."""
        if level == "national":
            shift, ratio = np.zeros(len(doy)), np.ones(len(doy))
        elif level == "province":
            shift, ratio = self.shift_province[province], self.ratio_province[province]
        else:
            shift, ratio = self.shift_county[county], self.ratio_county[county]
        if not use_shift:
            shift = np.zeros(len(doy))
        if not use_ratio:
            ratio = np.ones(len(doy))
        day = np.clip(np.rint(doy - shift), 1, 366).astype(int)
        return np.clip(ratio * self.hazard[day], 1e-4, 1 - 1e-4)


def fit(doy: np.ndarray, positive: np.ndarray, county: np.ndarray, province_of_county: np.ndarray,
        n_province: int, bandwidth: float = BANDWIDTH, prior: float = PRIOR_EVENTS) -> LocalCalendar:
    """Fit the three levels from at-risk survey units.

    doy, positive, county are per unit; county holds integer codes and
    province_of_county maps each county code to its province code.
    """
    pos = positive.astype(float)
    n_county = len(province_of_county)
    events = np.bincount(doy, weights=pos, minlength=367)
    units = np.bincount(doy, minlength=367).astype(float)
    kernel = _kernel(bandwidth)
    hazard = (kernel @ events) / np.maximum(kernel @ units, 1e-9)
    province = province_of_county[county]

    def level_shift(group, n_group, parent_mean):
        n = np.bincount(group, weights=pos, minlength=n_group)
        total = np.bincount(group, weights=pos * doy, minlength=n_group)
        return (total + prior * parent_mean) / (n + prior)

    def level_ratio(group, n_group, shift_per_unit, parent_ratio):
        day = np.clip(np.rint(doy - shift_per_unit), 1, 366).astype(int)
        observed = np.bincount(group, weights=pos, minlength=n_group)
        expected = np.bincount(group, weights=hazard[day], minlength=n_group)
        return (observed + prior * parent_ratio) / (expected + prior)

    national_mean = (pos * doy).sum() / max(pos.sum(), 1.0)
    mean_p = level_shift(province, n_province, national_mean)
    shift_p = mean_p - national_mean
    ratio_p = level_ratio(province, n_province, shift_p[province], 1.0)
    mean_c = level_shift(county, n_county, mean_p[province_of_county])
    shift_c = mean_c - national_mean
    ratio_c = level_ratio(county, n_county, shift_c[county], ratio_p[province_of_county])
    return LocalCalendar(hazard, shift_p, ratio_p, shift_c, ratio_c)
