#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""catalog.py

Methods for generating Scenes from catalogs and vice versa.
Also defines the standard source catalog data type.
"""


import numpy as np
from forcepho.sources import Scene, Galaxy


__all__ = ["sourcecat_dtype", "rectify_catalog",
           "scene_to_catalog", "catalog_to_scene",
           "SHAPE_COLS", "FLUX_COL", "PAR_COLS"]


# name of GPU relevant parameters in the source catalog
SHAPE_COLS = ["ra", "dec", "q", "pa", "nsersic", "rhalf"]
FLUX_COL = "flux"
PAR_COLS = ["id"] + SHAPE_COLS + [FLUX_COL]


def sourcecat_dtype(source_type=np.float64, bands=None):
    """Get a numpy.dtype object that describes the structured array
    that will hold the source parameters
    """
    nband = len(bands)
    tags = ["id", "source_index", "is_active", "is_valid", "n_iter", "n_patch"]

    dt = [(t, np.int32) for t in tags]
    dt += [(c, source_type)
           for c in SHAPE_COLS]
    dt += [(c, source_type, nband)
           for c in [FLUX_COL]]
    return np.dtype(dt)


def scene_to_catalog(scene, band_ids, cat_dtype):
    """Convert a scene to a structured ndarray of parameters.

    Parameters
    -----------
    scene : Scene() instance
        The scene, containing a list of sources

    band_ids : list of ints or slice
        The elements of the `"flux"` column in output catalog corresponding to
        the the `flux` vector attribute of each source in the scene.
    """
    active = np.zeros(nactive, dtype=cat_dtype)
    for i, row in enumerate(nactive):
        s = scene.sources[i]
        pars = s.ra, s.dec, s.q, s.pa, s.sersic, s.rh
        for j, f in enumerate(SHAPE_COLS):
            active[i][f] = pars[j]
        active[i][FLUX_COL][band_ids] = s.flux
    return active


def catalog_to_scene(sourcepars, band_ids, filters,
                     splinedata=None, free_sersic=True):
    """Build a scene from a structured array of source parameters including
    fluxes through a set of filters.

    Parameters
    ---------
    sourcepars : structured ndarray
        each row is a source.  The relevant columns are described by
        SHAPE_COLS and FLUX_COL, and it should have an "id" column.

    band_ids : list of ints or slice
        The elements of the flux array in `sourcepars` corresponding to
        the given filters.

    filters : list of strings
        The list of the band names that are being used for this scene.

    splinedata : string
        Path to the HDF5 file containing spline information.
        This should be the actual spline information...

    Returns
    -------
    scene: Scene object
    """
    #sourcepars = sourcepars.astype(np.float)
    # get all sources
    sources = []
    for ii, pars in enumerate(sourcepars):
        x, y, q, pa, n, rh = [pars[f] for f in SHAPE_COLS]
        gid = pars["id"]
        flux = pars[FLUX_COL]
        s = Galaxy(filters=filters, splinedata=splinedata,
                   free_sersic=free_sersic)
        s.global_id = gid
        s.sersic = n
        s.rh = np.clip(rh, 0.05, 0.20)
        s.flux = flux[band_ids]
        s.ra = x
        s.dec = y
        s.q = np.clip(q, 0.2, 0.9)
        s.pa = pa
        sources.append(s)

    # generate scene
    scene = Scene(sources)

    return(scene)


def rectify_catalog(sourcecatfile, rhrange=(0.03, 0.3), rotate=False, reverse=True):
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
    bands = [b.upper() for b in header["FILTERS"].split(",")]

    n_sources = len(cat)
    cat_dtype = sourcecat_dtype(bands=bands)
    sourcecat = np.zeros(n_sources, dtype=cat_dtype)
    sourcecat["source_index"][:] = np.arange(n_sources)
    for f in cat.dtype.names:
        if f in sourcecat.dtype.names:
            sourcecat[f][:] = cat[f][:]
    # --- Rectify shape columns ---
    sourcecat["nsersic"] = 3.0  # middle of range
    bad = ~np.isfinite(sourcecat["rhalf"])
    sourcecat["rhalf"][bad] = rhrange[0]
    sourcecat["rhalf"][:] = np.clip(sourcecat["rhalf"], *rhrange)
    # rotate PA by +90 degrees but keep in the interval [-pi/2, pi/2]
    if rotate:
        p = sourcecat["pa"] > 0
        sourcecat["pa"] += np.pi / 2. - p * np.pi
    if reverse:
        sourcecat["pa"] *= -1.0

    return sourcecat, bands, header
