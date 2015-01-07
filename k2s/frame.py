# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Frame"]

import numpy as np
from simplexy import simplexy


class Frame(object):

    def __init__(self, img, err_img, mask=None, **kwargs):
        if mask is None:
            self.mask = np.isfinite(img)
        else:
            self.mask = mask
        self.img = img
        self.err_img = err_img

        # Compute the pixel positions.
        self.shape = self.img.shape
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]),
                           indexing="ij")
        self.xi = np.array(x[self.mask], dtype=np.int)
        self.x = np.array(self.xi, dtype=np.float64)
        self.yi = np.array(y[self.mask], dtype=np.int)
        self.y = np.array(self.yi, dtype=np.float64)

        # Initialize the coordinate set.
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        if not np.any(self.mask):
            self.coords = []
            return
        tmp = np.array(self.img)
        tmp[~(self.mask)] = np.median(tmp[self.mask])
        try:
            self.coords = simplexy(tmp, **kwargs)
        except RuntimeError:
            self.coords = []

    def __len__(self):
        return len(self.coords) if self.coords is not None else 0
