from __future__ import print_function, division

import os
import glob
import fitsio
import numpy as np
from simplexy import simplexy
# import matplotlib.pyplot as pl
from multiprocessing import Pool


dt = np.dtype([("time", np.float32), ("flux", np.float32), ("bkg", np.float32),
               ("x", np.float32), ("y", np.float32), ("quality", np.int32)])


def process_file(fn):
    # Skip short cadence targets.
    if "spd" in fn:
        return

    # Construct the output filename.
    pre, post = os.path.split(fn)
    a, b, _ = post.split("-")
    outfn = os.path.join("lightcurves", pre[len("data/"):],
                         a+"-"+b+"-lc.fits.gz")

    # Don't overwrite.
    if os.path.exists(outfn):
        return

    print("{0} -> {1}".format(fn, outfn))

    # Read the data.
    data, hdr = fitsio.read(fn, columns=["TIME", "FLUX", "QUALITY"],
                            header=True)
    table = np.empty(len(data["TIME"]), dtype=dt)
    for k in ["x", "y", "flux", "bkg"]:
        table[k] = np.nan

    table["time"] = data["TIME"]
    table["quality"] = data["QUALITY"]
    m = np.isfinite(table["time"])
    print(np.min(table["time"][m]), np.max(table["time"][m]))

    for i, frame in enumerate(data["FLUX"]):
        # Run simplexy on the frame.
        frame[frame == 0.0] = np.nan
        try:
            result = simplexy(frame)
        except RuntimeError:
            continue
        if not len(result):
            continue

        # Save the brightest source to the results table.
        row = result[0]
        for k in row.dtype.names:
            table[k][i] = row[k]

    # Save the output file.
    try:
        os.makedirs(os.path.split(outfn)[0])
    except os.error:
        pass
    fitsio.write(outfn, table, clobber=True, header=hdr)

    # pl.clf()
    # pl.plot(table["time"], table["flux"], ".k")
    # pl.savefig(outfn + ".png")


filenames = glob.glob("data/c0/202000000/59000/*.fits.gz")
# filenames = glob.glob("data/c1/*/*/*.fits.gz")
pool = Pool()
pool.map(process_file, filenames)
