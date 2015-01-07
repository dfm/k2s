from __future__ import print_function, division

import os
import glob
import fitsio
import numpy as np
from multiprocessing import Pool

from k2s import TimeSeries


dt = np.dtype([("cadenceno", np.int32), ("time", np.float32),
               ("timecorr", np.float32), ("pos_corr1", np.float32),
               ("pos_corr2", np.float32), ("quality", np.int32),
               ("flux", np.float32), ("bkg", np.float32),
               ("x", np.float32), ("y", np.float32)])


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
    data, hdr = fitsio.read(fn,
                            header=True)
    table = np.empty(len(data["TIME"]), dtype=dt)

    # Initialize the new columns to NaN.
    for k in ["x", "y", "flux", "bkg"]:
        table[k] = np.nan

    # Copy across the old columns.
    for k in ["cadenceno", "time", "timecorr", "pos_corr1", "pos_corr2",
              "quality"]:
        table[k] = data[k.upper()]

    print(data.dtype)
    ts = TimeSeries(data["TIME"], data["FLUX"], data["FLUX_ERR"],
                    data["QUALITY"])
    for i, frame in enumerate(ts.frames):
        if not len(frame):
            continue
        # Save the brightest source to the results table.
        row = frame.coords[0]
        for k in row.dtype.names:
            table[k][i] = row[k]

    # Save the output file.
    try:
        os.makedirs(os.path.split(outfn)[0])
    except os.error:
        pass
    fitsio.write(outfn, table, clobber=True, header=hdr)


filenames = glob.glob("data/c0/202000000/59000/*.fits.gz")
# filenames = glob.glob("data/c1/*/*/*.fits.gz")
pool = Pool()
pool.map(process_file, filenames)
