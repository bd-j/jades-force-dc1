#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, glob
import numpy as np
import matplotlib.pyplot as pl

import h5py


def split_patch_exp(patch):
    pixdat = ["xpix", "ypix", "data", "ierr"]
    splits = [np.split(getattr(patch, arr), np.cumsum(patch.exposure_N)[:-1])
              for arr in pixdat]

    return splits


def show_exp(xpix, ypix, value, ax=None, **imshow_kwargs):
    """Create a rectangular image that bounds the given pixel coordinates
    and assign `value` to the correct pixels. Pixels in the rectangle that do
    not have assigned values are given nan.  use imshow to display the image in
    standard astro format (x increasing left to right, y increasing bottom to
    top)
    """
    lo = np.array((xpix.min(), ypix.min())) - 0.5
    hi = np.array((xpix.max(), ypix.max())) + 0.5
    size = hi - lo
    im = np.zeros(size.astype(int)) + np.nan

    x = (xpix-lo[0]).astype(int)
    y = (ypix-lo[1]).astype(int)
    # This is the correct ordering of xpix, ypix subscripts
    im[x, y] = value

    ax.imshow(im.T, origin="lower",
              extent=(lo[0], hi[0], lo[1], hi[1]),
              **imshow_kwargs)
    return ax


def sky_to_pix(ra, dec, group=None, patch=None, exp_idx=0, ref_coords=0.):

    if group:
        crval = group["crval"][:]
        crpix = group["crpix"][:]
        CW = group["CW"][:]
    elif patch:
        crval = patch.crval[exp_idx]
        crpix = patch.crpix[exp_idx]
        CW = patch.CW[exp_idx]
        ref_coords = patch.patch_reference_coordinates

    if len(CW) != len(ra):
        CW = CW[0]

    sky = np.array([ra, dec]).T - (crval + ref_coords)
    pix = np.matmul(CW, sky[:, :, None])[..., 0] + crpix

    return pix


def mark_sources(ra, dec, group, ref_coords=0, ax=None,
                 plot_kwargs={"marker": "x", "linestyle": "", "color": "red"},
                 **extras):

    pix = sky_to_pix(ra, dec, group, ref_coords=ref_coords)

    plot_kwargs.update(extras)
    ax.plot(pix[:, 0], pix[:, 1], **plot_kwargs)
    return ax


def show_patch(fn, exposure_inds=[0, -1], show_fixed=True, show_active=False,
               imshow_kwargs={"vmin": -0.1, "vmax": 0.5}, **extras):

    disk = h5py.File(fn, "r")
    epaths = disk["epaths"][:]
    try:
        active = disk["active"][:]
    except(KeyError):
        active = None
    try:
        fixed = disk["fixed"][:]
    except(KeyError):
        fixed = None
    try:
        ref = disk["reference_coordinates"][:]
    except(KeyError):
        ref = np.array([np.median(active["ra"]), np.median(active["dec"])])

    vtypes = ["data", "active_residual"]
    ne = len(exposure_inds)

    fig, axes = pl.subplots(ne, 3, sharex="row", sharey="row")
    for i, e in enumerate(exposure_inds):
        epath = epaths[e]
        g = disk[epath]
        model = g["data"][:] - g["active_residual"][:]
        for j, vtype in enumerate(vtypes):
            ax = axes[i, j]
            show_exp(g["xpix"][:], g["ypix"][:], g[vtype][:], ax=ax, **imshow_kwargs)
            #ax.set_title(e)
        ax = axes[i, -1]
        show_exp(g["xpix"][:], g["ypix"][:], model, ax=ax, **imshow_kwargs)
        ee = epath.decode("utf")
        axes[i, 0].set_ylabel(" ".join(ee.replace(".flx", "").split("_")[-3:]))

        if show_active and (active is not None):
            ax = mark_sources(active["ra"], active["dec"], g,
                              ref_coords=ref, ax=ax, color="red")
        if show_fixed and (fixed is not None):
            for j in range(3):
                ax = axes[i, j]
                ax = mark_sources(fixed["ra"], fixed["dec"], g,
                                  ref_coords=ref, ax=ax, color="magenta")

    titles = ["data", "data-model", "model"]
    [ax.set_title(t) for ax, t in zip(axes[0, :], titles)]
    ti = "Center index = {}\n ra, dec=({:10.7f}, {:10.7f})"
    if active is not None:
        fig.suptitle(ti.format(active[0]["source_index"], active[0]["ra"], active[0]["dec"]))
    pl.show()
    return fig, axes, disk


if __name__ == "__main__":

    fn = glob.glob(sys.argv[1])[0]
    fig, axes, data = show_patch(fn)