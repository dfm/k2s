# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["TimeSeries"]

import numpy as np
from itertools import izip

from .frame import Frame


class TimeSeries(object):

    def __init__(self, time, flux_images, ferr_images, quality, **kwargs):
        # Initialize the frame images.
        self.time = time
        self.frames = []
        for i, (f, fe) in enumerate(izip(flux_images, ferr_images)):
            frame = []
            if quality[i] == 0:
                frame = Frame(f, fe, **kwargs)
                if not np.any(frame.mask):
                    frame = []
            self.frames.append(frame)

        # Update the frames to have a coherent time series.
        self.initialize()

    def initialize(self):
        # Traverse the graph and construct the (greedy) best path.
        ns = min(map(len, filter(len, self.frames)))
        metric = np.array([1.0, 1.0, 1e-8])
        current = None
        for t, node in enumerate(self.frames):
            if not len(node):
                continue
            if current is None:
                current = node.coords[:ns]
                node.coords = current
                continue

            # Compute the set of distances between this node and the current
            # position.
            r = sum([(node.coords[k][:, None] - current[k][None, :]) ** 2
                     * metric[i] for i, k in enumerate(("x", "y", "flux"))])
            r0 = np.array(r)

            # Loop over the permutations and greedily choose the best update.
            rows, cols = np.arange(r.shape[0]), np.arange(r.shape[1])
            update = np.nan + np.zeros(ns)
            while any(np.isnan(update)):
                row, col = np.unravel_index(np.argmin(r), r.shape)
                update[cols[col]] = rows[row]
                r = np.delete(r, row, axis=0)
                r = np.delete(r, col, axis=1)
                rows = np.delete(rows, row)
                cols = np.delete(cols, col)
            update = np.array(update, dtype=int)

            # Compute the total cost. MAGIC
            cost = np.sum(r0[(update, range(ns))])
            if cost > 10.0:
                node.coords = None
                continue

            # Update the current locations.
            current = np.array([node.coords[j] for j in update])
            self.frames[t].coords = current
