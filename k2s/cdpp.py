# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["compute_cdpp"]

import numpy as np

DEFAULT_CADENCE = 1626.0 / 60. / 60.  # In hours.


def compute_cdpp(time, flux, window, cadence=DEFAULT_CADENCE, robust=False):
    # Mask missing data and fail if no acceptable points exist.
    m = np.isfinite(time) * np.isfinite(flux)
    if not np.sum(m):
        return np.inf
    t, f = time[m], flux[m]

    # Compute the running relative std deviation.
    std = np.empty(len(t))
    hwindow = 0.5 * window
    for i, t0 in enumerate(t):
        m = np.abs(t - t0) < hwindow
        if np.sum(m) <= 0:
            std[i] = np.inf
        if robust:
            mu = np.median(f[m])
            std[i] = np.sqrt(np.median((f[m] - mu) ** 2)) / mu
        else:
            std[i] = np.std(f[m]) / np.mean(f[m])

    # Normalize by the window size.
    return 1e6 * np.median(std) / np.sqrt(window / cadence)
