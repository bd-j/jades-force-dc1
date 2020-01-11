#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from forcepho.sources import Scene, Galaxy


# name of GPU relevant parameters in the source catalog
SHAPE_COLS = ["ra", "dec", "q", "pa", "nsersic", "rhalf"]
FLUX_COL = "flux"
PAR_COLS = ["id"] + SHAPE_COLS + FLUX_COL


def sourcecat_dtype(source_type=np.float64, bands=None):
    nband = len(bands)
    tags = ["id", "source_index", "is_active", "is_valid", "n_iter", "n_patch"]

    dt = [(t, np.int32) for t in tags]
    dt += [(c, source_type)
           for c in SHAPE_COLS]
    dt += [(c, source_type, nband)
           for c in [FLUX_COL]]
    return np.dtype(dt)


def scene_to_catalog(scene, band_ids, cat_dtype=None):
    active = np.zeros(nactive, dtype=cat_dtype)
    for i, row in enumerate(nactive):
        s = scene.sources[i]
        pars = s.ra, s.dec, s.q, s.pa, s.nsersic, s.rh
        for j, f in enumerate(SHAPE_COLS):
            active[i][f] = pars[j]
        active[i][FLUX_COL][band_ids] = s.flux
    return active


def catalog_to_scene(sourcepars, band_ids, filters,
                     splinedata=None, free_sersic=True):
    """Build a scene from a set of source parameters and fluxes through a
    set of filters.

    Parameters
    ---------
    sourcepars : structured ndarray
        each row is a source.  It should unpack as:
        id, ra, dec, q, pa, n, rh, flux, flux_unc

    band_ids : list of ints or slice
        The elements of the flux array in `sourcepars` corresponding to
        the given filters.

    filters : list of strings
        The list of the band names that are being used for this patch

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
        x, y, q, pa, n, rh, flux = [pars[f] for f in SHAPE_COLS]
        gid = pars["id"]
        flux = pars[FLUX_COL]
        s = Galaxy(filters=filters, splinedata=splinedata,
                    free_sersic=free_sersic)
        s.global_id = gid
        s.sersic = n
        s.rh = np.clip(rh, 0.05, 0.10)
        s.flux = flux[band_ids]
        s.ra = x
        s.dec = y
        s.q = np.clip(q, 0.2, 0.9)
        s.pa = np.deg2rad(pa)
        sources.append(s)

    # generate scene
    scene = Scene(sources)

    return(scene)
