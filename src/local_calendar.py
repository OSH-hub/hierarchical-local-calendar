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
    hazard: np.ndarray
    shift_province: np.ndarray
    ratio_province: np.ndarray
    shift_county: np.ndarray
    ratio_county: np.ndarray

    def predict(self, doy: np.ndarray, county: np.ndarray, province: np.ndarray, level: str = "county",
                use_shift: bool = True, use_ratio: bool = True) -> np.ndarray:
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
