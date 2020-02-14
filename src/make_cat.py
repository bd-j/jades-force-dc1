#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, os, sys
import numpy as np

import h5py
from astropy.io import fits
from astropy.wcs import WCS


SHAPE_COLS = ["ra", "dec", "q", "pa", "nsersic", "rhalf"]


def make_chaincat(filename):

    with h5py.File(filename, "r") as disk:
        chain = disk["chain"][:]
        bands = disk["bandlist"][:]
        ref = disk["reference_coordinates"][:]
        active = disk["active"][:]

    bands = [b.decode("utf-8") for b in bands]

    # --- Get sizes of things ----
    n_iter, n_param = chain.shape
    n_band = len(bands)

    n_param_per_source = n_band + 6
    assert (np.mod(n_param, n_param_per_source) == 0)
    n_source = int(n_param / n_param_per_source)

    # --- generate dtype ---
    colnames = bands + SHAPE_COLS
    cols = [("id", np.int)] + [(c, np.float, (n_iter,)) for c in colnames]
    dtype = np.dtype(cols)

    # --- make and fill catalog
    cat = np.zeros(n_source, dtype=dtype)
    for s in range(n_source):
        for j, col in enumerate(colnames):
            cat[s][col] = chain[:, s * n_param_per_source + j]

    # rectify parameters
    cat["ra"] += ref[0]
    cat["dec"] += ref[1]
    cat["q"] = cat["q"]**2
    cat["pa"] = -np.rad2deg(cat["pa"])

    cat["id"] = active["source_index"]
    # document units and number of iterations
    # units: image_units, degrees, degrees, b/a, degrees E of North, sersic index, arcsec
    # n_iter
    return cat


def summary_cat(chaincat, estimate=np.mean, wcs=None):
    """Return point estimates and uncertaites for all parameters in a given
    chaincat
    """
    efmt = "{}_unc"

    colnames = list(chaincat.dtype.names)
    _ = colnames.pop(colnames.index("id"))
    allnames = colnames + [efmt.format(c) for c in colnames]
    dtype = [("id", np.int)] + [(c, np.float) for c in allnames]

    if wcs is not None:
        x, y = wcs.all_world2pix(chaincat["ra"], chaincat["dec"], 1)
        chaincat["ra"] = x
        chaincat["dec"] = y

    cat = np.zeros(len(chaincat), dtype=dtype)
    for i, row in enumerate(chaincat):
        for j, c in enumerate(colnames):
            cat[i][c] = estimate(row[c])
            cat[i][efmt.format(c)] = np.std(row[c])

    return cat


if __name__ == "__main__":

    wcs = None
    imname = os.path.expandvars("$HOME/Projects/jades_force/data/galsim/vrfnq_v0_sci.fits")
    if imname is not None:
        wcs = WCS(fits.getheader(imname))
    search = "output/*patchid??.h5"
    files = glob.glob(search)

    chaincats = [make_chaincat(f) for f in files]
    summaries = [summary_cat(chains, wcs=wcs) for chains in chaincats]

    # get the source indicies of each row
    inds = [chains["id"] for chains in chaincats]
    order = np.argsort(np.concatenate(inds))

    summary = np.concatenate(summaries)[order]
    summary["id"] = np.concatenate(inds)[order]
    fits.writeto("summary_galsim_v0.fits", summary, overwrite=True)

    chains = [c[0] for c in chaincats]
    chaincat = np.concatenate(chains)[order]

    fits.writeto("chains_galsim_v0.fits", chaincat, overwrite=True)