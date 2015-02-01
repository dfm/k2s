from __future__ import print_function, division

import os
import glob
import numpy as np
import matplotlib.pyplot as pl
from multiprocessing import Pool

from astropy.io import fits
from astropy.wcs import WCS

from k2s import compute_cdpp

apertures = np.arange(0.5, 5.5, 0.5)
dt = np.dtype([("cadenceno", np.int32), ("time", np.float32),
               ("timecorr", np.float32), ("pos_corr1", np.float32),
               ("pos_corr2", np.float32), ("quality", np.int32),
               ("flux", (np.float32, len(apertures))),
               ("bkg", (np.float32, len(apertures)))])
dt2 = np.dtype([("radius", np.float32), ("cdpp6", np.float32)])


def process_file(fn):
    # Skip short cadence targets.
    if "spd" in fn:
        return

    # Construct the output filename.
    pre, post = os.path.split(fn)
    a, b, _ = post.split("-")
    outfn = os.path.join("lightcurves", pre[len("data/"):],
                         a+"-"+b+"-lc.fits")

    # Don't overwrite.
    if os.path.exists(outfn):
        return

    # Read the data.
    hdus = fits.open(fn)
    data = hdus[1].data
    table = np.empty(len(data["TIME"]), dtype=dt)

    # Initialize the new columns to NaN.
    for k in ["flux", "bkg"]:
        table[k] = np.nan

    # Copy across the old columns.
    for k in ["cadenceno", "time", "timecorr", "pos_corr1", "pos_corr2",
              "quality"]:
        table[k] = data[k.upper()]

    # Use the WCS to find the center of the star.
    hdr = hdus[2].header
    wcs = WCS(hdr)
    cy, cx = wcs.wcs_world2pix(hdr["RA_OBJ"], hdr["DEC_OBJ"], 0.0)

    # Choose the set of apertures.
    aps = []
    shape = data["FLUX"][0].shape
    xi, yi = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
    for r in apertures:
        r2 = r*r
        aps.append((xi - cx) ** 2 + (yi - cy) ** 2 < r2)

    # Loop over the frames and do the aperture photometry.
    for i, img in enumerate(data["FLUX"]):
        fm = np.isfinite(img)
        fm[fm] = img[fm] > 0.0
        if not np.any(fm):
            continue
        for j, mask in enumerate(aps):
            # Choose the pixels in and out of the aperture.
            m = mask * fm
            bgm = (~mask) * fm

            # Skip if there are no good pixels in the aperture.
            if not np.any(m):
                continue

            # Estimate the background and flux.
            if np.any(bgm):
                bkg = np.median(img[bgm])
            else:
                bkg = np.median(img[frame.mask])
            table["flux"][i, j] = np.sum(img[m] - bkg)
            table["bkg"][i, j] = bkg

    # Compute the number of good times.
    nt = int(np.sum(np.any(np.isfinite(table["flux"]), axis=1)))
    print("{0} -> {1} ; {2}".format(fn, outfn, nt))

    # Skip it if there aren't *any* good times.
    if nt == 0:
        return

    # Save the aperture information and precision.
    ap_info = np.empty(len(apertures), dtype=dt2)
    for i, r in enumerate(apertures):
        ap_info[i]["radius"] = r

        # Compute the precision.
        t, f = table["time"], table["flux"][:, i]
        # ap_info[i]["cdpp3"] = compute_cdpp(t, f, 3.)
        ap_info[i]["cdpp6"] = compute_cdpp(t, f, 6.)
        # ap_info[i]["cdpp12"] = compute_cdpp(t, f, 12.)

    try:
        os.makedirs(os.path.split(outfn)[0])
    except os.error:
        pass
    hdr = hdus[1].header
    hdr["CEN_X"] = float(cx)
    hdr["CEN_Y"] = float(cy)
    hdus_out = fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.BinTableHDU.from_columns(table, header=hdr),
        fits.BinTableHDU.from_columns(ap_info),
    ])
    hdus_out.writeto(outfn, clobber=True)

    hdus.close()
    hdus_out.close()


def wrap(fn):
    try:
        process_file(fn)
    except:
        print("failure: {0}".format(fn))
        import traceback
        traceback.print_exc()
        raise

# filenames = glob.glob("data/c1/201500000/85000/ktwo201585079-c01_lpd-targ.fits.gz")
# map(process_file, filenames)
# assert 0

filenames = glob.glob("data/c1/*/*/*.fits.gz")
pool = Pool()
pool.map(wrap, filenames)
