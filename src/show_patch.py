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
    lo = np.array((xpix.min(), ypix.min()))
    hi = np.array((xpix.max(), ypix.max()))
    size = hi - lo + 1
    im = np.zeros(size.astype(int)) + np.nan

    x = (xpix-lo[0]).astype(int)
    y = (ypix-lo[1]).astype(int)
    # This is the correct ordering of xpix, ypix subscripts
    im[x, y] = value

    ax.imshow(im.T, origin="lower",
              extent=(lo[0], hi[0], lo[1], hi[1]),
              **imshow_kwargs)
    return ax


if __name__ == "__main__":

    fn = glob.glob(sys.argv[1])[0]

    disk = h5py.File(fn, "r")
    epaths = disk["epaths"][:]
    vtypes = ["data", "active_residual"]
    showpaths = epaths[0], epaths[-1]

    ne = 2
    vrange = (-0.5, 0.5)

    fig, axes = pl.subplots(ne, 3, sharex="row", sharey="row")
    for i, e in enumerate(showpaths):
        g = disk[e]
        model = g["data"][:] - g["active_residual"][:]
        for j, vtype in enumerate(vtypes):
            ax = axes[i, j]
            show_exp(g["xpix"][:], g["ypix"][:], g[vtype][:], ax=ax, vmin=-0.5, vmax=0.5)
            #ax.set_title(e)
        ax = axes[i, -1]
        show_exp(g["xpix"][:], g["ypix"][:], model, ax=ax, vmin=-0.1, vmax=0.5)
        ee = e.decode("utf")
        axes[i, 0].set_ylabel(" ".join(ee.replace(".flx", "").split("_")[-3:]))

    active = disk["active"][:]
    titles = ["data", "data-model", "model"]
    [ax.set_title(t) for ax, t in zip(axes[0, :], titles)]
    fig.suptitle("Center index = {}".format(active[0]["source_index"]))
    pl.show()