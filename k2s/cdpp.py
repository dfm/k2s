# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["compute_cdpp"]

import numpy as np

DEFAULT_CADENCE = 1626.0 / 60. / 60.  # In hours.


def compute_cdpp(time, flux, window, cadence=DEFAULT_CADENCE, robust=False):
    hwindow = 0.5 * window
    m = np.isfinite(time) * np.isfinite(flux)
    t, f = time[m], flux[m]
    std = np.empty(len(t))
    for i, t0 in enumerate(t):
        m = np.abs(t - t0) < hwindow
        if robust:
            mu = np.median(f[m])
            std[i] = np.sqrt(np.median((f[m] - mu) ** 2)) / mu
        else:
            std[i] = np.std(f[m]) / np.mean(f[m])
    return 1e6 * np.median(std) / np.sqrt(window / cadence)
