#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""catalog.py

Methods for generating Scenes from catalogs and vice versa.
Also defines the standard source catalog data type.
"""


import numpy as np


__all__ = ["sourcecat_dtype", "rectify_catalog",
           "SHAPE_COLS", "FLUX_COL", "PAR_COLS"]


# name of GPU relevant parameters in the source catalog
SHAPE_COLS = ["ra", "dec", "q", "pa", "sersic", "rhalf"]
FLUX_COL = "flux"
PAR_COLS = ["id"] + [FLUX_COL] + SHAPE_COLS


def sourcecat_dtype(source_type=np.float64, bands=None):
    """Get a numpy.dtype object that describes the structured array
    that will hold the source parameters
    """
    nband = len(bands)
    tags = ["id", "source_index", "is_active", "is_valid", "n_iter", "n_patch"]

    dt = [(t, np.int32) for t in tags]
    dt += [(c, source_type)
           for c in SHAPE_COLS]
    dt += [(c, source_type)
           for c in bands]
    return np.dtype(dt)


def rectify_catalog(sourcecatfile, rhrange=(0.05, 0.25), qrange=(0.2, 0.99),
                    rotate=False, reverse=True):
    """Read the given catalog file and generate a `sourcecat` structured
    ndarray, which is an ndarray matched row-by-row but has all required
    columns.  Also forces parameters to be in valid ranges with valid formats

    Parameters
    ----------
    sourceatfile : string
        Path to the FITS binary table representing the initilization
        catalog. This file must have a header entry "FILTERS" giving a
        comma-separated list of bands in the same order as the "flux"
        column.

    rhrange : 2-tuple of floats, optional (default: 0.03, 0.3)
        The range of half-light radii that is acceptable in arcsec.
        Input radii will be clipped to this range.

    rotate : bool, optional, default=False
        Whether to rotate the PA by 90 degrees

    reverse : bool, optional, default=True
        Whether to reverse the direction of the PA (i.e. from CW to CCW)

    Returns
    -------
    sourcecat : structured ndarray of shape (n_sources,)

    bands : list of strings

    header : astropy header object

    """
    from astropy.io import fits
    cat = fits.getdata(sourcecatfile)
    header = fits.getheader(sourcecatfile)
    bands = [b.upper().strip() for b in header["FILTERS"].split(",")]

    n_sources = len(cat)
    cat_dtype = sourcecat_dtype(bands=bands)
    sourcecat = np.zeros(n_sources, dtype=cat_dtype)
    sourcecat["source_index"][:] = np.arange(n_sources)
    for f in cat.dtype.names:
        if f in sourcecat.dtype.names:
            sourcecat[f][:] = cat[f][:]

    for i, b in enumerate(bands):
        sourcecat[b][:] = cat["flux"][:, i]

    # --- Rectify shape columns ---
    sourcecat["sersic"] = 3.0  # middle of range
    bad = ~np.isfinite(sourcecat["rhalf"])
    sourcecat["rhalf"][bad] = rhrange[0]
    sourcecat["rhalf"][:] = np.clip(sourcecat["rhalf"], *rhrange)
    sourcecat["q"][:] = np.clip(sourcecat["q"], *qrange)
    # rotate PA by +90 degrees but keep in the interval [-pi/2, pi/2]
    if rotate:
        p = sourcecat["pa"] > 0
        sourcecat["pa"] += np.pi / 2. - p * np.pi
    if reverse:
        sourcecat["pa"] *= -1.0

    return sourcecat, bands, header
