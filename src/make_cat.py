#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, os, sys
import numpy as np

import h5py
from astropy.io import fits
from astropy.wcs import WCS


SHAPE_COLS = ["ra", "dec", "q", "pa", "nsersic", "rhalf"]


def fixed_aperture_fraction(nsersic, rhalf, rap):
    """Compute the fraction of the flux that falls within a given radius of
    the source center for a _circularized_ shape.  The circularization means
    this is not the same as the fraction of flux of a galaxy that would fall
    within a circular aperture placed on the galaxy image (which depends on q as
    well)

    Parameters
    ----------
    nsersic : float
        The sersic index

    rhalf : float
        The half-light radius

    rap : float
        The aperture radius, in the same units as `rhalf` (usually arcsec)

    Returns
    -------
    frac : float
        The fraction of the total light that is enclosed within `rap`
    """
    from scipy.special import gammainc, gammaincinv
    n = nsersic
    # note gammaincinv is for the normalized gamma function, so...
    b_n = gammaincinv(2 * n, 0.5)
    x = b_n * (rap / rhalf)**(1./n)
    frac = gammainc(2 * n, x)
    return frac


def make_chaincat(filename, apertures=[]):
    """Make a catalog from the chain.  This essentially names the columns in
    the `chain` dataset of the provided file and makes several transformations:
    * ra           -> ra + reference_ra
    * dec          -> dec + refeence_dec
    * sqrt(a/b)    -> q
    * pa (radians) -> pa (degrees)

    Parameters
    ----------
    apertures : list of float, optional (default: [])
       A list of aperture radii (in same units as rhalf).  Note these are for "circularized" profiles
    """
    with h5py.File(filename, "r") as disk:
        chain = disk["chain"][:]
        active = disk["active"][:]
        try:
            bands = disk["bandlist"][:]
            ref = disk["reference_coordinates"][:]
        except(KeyError):
            bands = disk.attrs["bandlist"][:]
            ref = disk.attrs["reference_coordinates"][:]

    bands = [b.decode("utf-8") for b in bands]

    aper_fmt = "{}_aper{:.0f}mas"
    aper_bands = []
    for r in apertures:
        aper_bands += [aper_fmt.format(b, r*1000) for b in bands]

    # --- Get sizes of things ----
    n_iter, n_param = chain.shape
    n_band = len(bands)

    n_param_per_source = n_band + 6
    assert (np.mod(n_param, n_param_per_source) == 0)
    n_source = int(n_param / n_param_per_source)

    # --- generate dtype ---
    colnames = bands + SHAPE_COLS
    cols = ([("id", np.int), ("x", np.float, (n_iter,)), ("y", np.float, (n_iter,))] +
            [(c, np.float, (n_iter,)) for c in colnames + aper_bands])
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

    if wcs is not None:
        
        x, y = wcs.all_world2pix(cat["ra"], cat["dec"], 1)
        cat["x"] = x
        cat["y"] = y

    # add aperture fluxes
    # this is dumb because the fractions are the same in each band
    # but it makes the logic for the summary catalog easier
    for r in apertures:
        frac = fixed_aperture_fraction(cat["nsersic"], cat["rhalf"], r)
        for b in bands:
            aper_name = aper_fmt.format(b, r*1000)
            cat[aper_name] = cat[b] * frac

    # document units and number of iterations
    # units: image_units, degrees, degrees, b/a, degrees E of North, sersic index, arcsec
    # n_iter
    return cat


def summary_cat(chaincat, estimate=np.mean, wcs=None):
    """Return point estimates and uncertaites for all parameters in a given
    chaincat

    Parameters
    ----------
    wcs : optional
        If given, use this WCS to convert celestial coordinates back into pixel coordinates.
    """
    efmt = "{}_unc"

    colnames = list(chaincat.dtype.names)
    _ = colnames.pop(colnames.index("id"))
    allnames = colnames + [efmt.format(c) for c in colnames]
    dtype = ([("id", np.int), ("patchid", np.int)] +
             [(c, np.float) for c in allnames])

    cat = np.zeros(len(chaincat), dtype=dtype)
    for i, row in enumerate(chaincat):
        for j, c in enumerate(colnames):
            cat[i][c] = estimate(row[c])
            cat[i][efmt.format(c)] = np.std(row[c])

    return cat


if __name__ == "__main__":

    wcs = None
    imname = os.path.expandvars("$HOME/Projects/jades_force/data/2019-mini-challenge/mosaics/st/trimmed/F200W_bkgsub.fits")
    wcs = WCS(fits.getheader(imname))

    pdir = os.path.expandvars("$HOME/Projects/jades_force/cannon/output/")
    search = os.path.join(pdir, "*[0-9].h5")
    files = glob.glob(search)
    patchid = [int(os.path.basename(f).split("idx")[-1].replace(".h5", ""))
               for f in files]

    chaincats = [make_chaincat(f, apertures=[0.10]) for f in files]

    # get the source indicies of each row
    inds = [chains["id"] for chains in chaincats]
    order = np.argsort(np.concatenate(inds))
    order = slice(None)

    chaincat = np.concatenate(chaincats)[order]
    fits.writeto("chains_mini-challenge-19_v0.fits", chaincat, overwrite=True)

    summaries = []
    for i, pid in enumerate(patchid):
        summary = summary_cat(chaincats[i], wcs=wcs)
        summary["patchid"] = pid
        summary["id"] = chaincats[i]["id"]
        summaries.append(summary)

    summary = np.concatenate(summaries)[order]
    fits.writeto("summary_mini-challenge-19_v0.fits", summary, overwrite=True)
