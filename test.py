from __future__ import print_function, division

import os
import glob
import fitsio
import numpy as np
from multiprocessing import Pool

from k2s import TimeSeries, compute_cdpp

apertures = np.arange(0.5, 10.5, 0.5)
dt = np.dtype([("cadenceno", np.int32), ("time", np.float32),
               ("timecorr", np.float32), ("pos_corr1", np.float32),
               ("pos_corr2", np.float32), ("quality", np.int32),
               ("flux", (np.float32, len(apertures))),
               ("bkg", (np.float32, len(apertures))),
               ("x", np.float32), ("y", np.float32)])
dt2 = np.dtype([("radius", np.float32), ("cdpp3", np.float32),
                ("cdpp6", np.float32), ("cdpp12", np.float32)])


def process_file(fn):
    # Skip short cadence targets.
    if "spd" in fn:
        return

    # Construct the output filename.
    pre, post = os.path.split(fn)
    a, b, _ = post.split("-")
    outfn = os.path.join("lightcurves", pre[len("data/"):],
                         a+"-"+b+"-lc.fits")

    # # Don't overwrite.
    # if os.path.exists(outfn):
    #     return

    print("{0} -> {1}".format(fn, outfn))

    # Read the data.
    data, hdr = fitsio.read(fn, header=True)
    table = np.empty(len(data["TIME"]), dtype=dt)

    # Initialize the new columns to NaN.
    for k in ["x", "y", "flux", "bkg"]:
        table[k] = np.nan

    # Copy across the old columns.
    for k in ["cadenceno", "time", "timecorr", "pos_corr1", "pos_corr2",
              "quality"]:
        table[k] = data[k.upper()]

    # This step actually does the photometry.
    ts = TimeSeries(data["TIME"], data["FLUX"], data["FLUX_ERR"],
                    data["QUALITY"])

    # Loop over the frames and copy over the output.
    for i, frame in enumerate(ts.frames):
        if not len(frame):
            continue
        # Save the brightest source to the results table.
        shape = frame.shape
        row = frame.coords[0]
        for k in ["x", "y"]:
            table[k][i] = row[k]

    # Find the median centroid of the star.
    m = np.isfinite(table["x"]) * np.isfinite(table["y"])
    cx, cy = np.median(table[m]["x"]), np.median(table[m]["y"])
    hdr["CENTROID_X"] = float(cx)
    hdr["CENTROID_Y"] = float(cy)

    # Choose the set of apertures.
    aps = []
    for r in apertures:
        r2 = r*r
        xi, yi = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
        aps.append((xi - cx) ** 2 + (yi - cy) ** 2 < r2)

    # Loop over the frames and do the aperture photometry.
    for i, frame in enumerate(ts.frames):
        if not len(frame):
            continue
        for j, mask in enumerate(aps):
            # Skip if there are no good pixels in the aperture.
            if not np.any(mask * frame.mask):
                continue

            # Estimate the background and flux.
            bkg = np.median(frame.img[(~mask) * frame.mask])
            table["flux"][i, j] = np.sum(frame.img[mask * frame.mask] - bkg)
            table["bkg"][i, j] = bkg

    print("{0} good times.".format(np.sum(np.all(np.isfinite(table["flux"]),
                                                 axis=1))))

    # Save the output file.
    try:
        os.makedirs(os.path.split(outfn)[0])
    except os.error:
        pass
    fitsio.write(outfn, table, clobber=True, header=hdr)

    # Save the aperture information and precision.
    ap_info = np.empty(len(apertures), dtype=dt2)
    for i, r in enumerate(apertures):
        ap_info[i]["radius"] = r

        # Compute the precision.
        t, f = table["time"], table["flux"][:, i]
        ap_info[i]["cdpp3"] = compute_cdpp(t, f, 3.)
        ap_info[i]["cdpp6"] = compute_cdpp(t, f, 6.)
        ap_info[i]["cdpp12"] = compute_cdpp(t, f, 12.)
    fitsio.write(outfn, ap_info)

    print(ap_info)

    import matplotlib.pyplot as pl
    m = np.isfinite(table["flux"][:, -1])
    mu = np.median(table["flux"][:, -1][m])
    pl.plot(table["time"], table["flux"][:, -1] / mu, ".k")
    # pl.plot(table["time"], table["bkg"], ".r")
    # pl.xlim(1940, 1975)
    pl.savefig("blah.png")


filenames = glob.glob("data/c0/202000000/86000/*202086255*.fits.gz")
map(process_file, filenames)
# print(filenames)
# filenames = glob.glob("data/*/*/*/*.fits.gz")
# pool = Pool()
# pool.map(process_file, filenames)
