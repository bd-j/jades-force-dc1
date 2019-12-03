#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_pdf import PdfPages

import h5py


def draw_ellipses(params, ax, cmap=get_cmap('viridis')):
    from matplotlib.patches import Ellipse

    ngauss = len(params)
    for i in range(ngauss):
        # need to swap axes here, not sure why
        mu = np.array([params[i]["x"], params[i]["y"]]) 
        vy = params[i]["vyy"]
        vx = params[i]["vxx"]
        vxy = params[i]["vxy"]
        # construct covar matrix and get eigenvalues
        S = np.array([[vx, vxy], [vxy, vy]])
        vals, vecs = np.linalg.eig(S)
        # get ellipse params
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=w, height=h, angle=theta)
        ax.add_artist(ell)
        #e.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ell.set_facecolor(cmap(params[i]["amp"]))
    return ax


def show(pname, width=100):
    with h5py.File(pname, "r") as pdat:
        m = pdat["models"][:]
        o = pdat["psf_image"][:]
        pars = pdat["parameters"][:]
        nfit, ngauss = pars.shape

    res = o - m
    sz = np.array(o.shape)
    s = ((sz - width) / 2).astype(int)
    p = ((sz + width) / 2).astype(int)
    sel = (slice(s[0], p[0]), slice(s[1], p[1]))
    gcmap = get_cmap('viridis')
    Z = [[0,0],[0,0]]
    levels = np.linspace(0, pars["amp"].max(), ngauss+1)
    dummy = pl.contourf(Z, levels, cmap=gcmap)

    pdf = PdfPages(pname.replace("h5", "pdf"))
    for ifit in range(nfit):
        fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)

        ax = axes[0, 0]
        d = ax.imshow(o[sel], origin='lower', cmap=gcmap)
        fig.colorbar(d, ax=ax)
        ax.text(0.1, 0.9, 'Truth', transform=ax.transAxes)

        ax = axes[0, 1]
        m1 = ax.imshow(m[ifit][sel], origin='lower', cmap=gcmap)
        fig.colorbar(m1, ax=ax)
        ax.text(0.1, 0.9, 'Model', transform=ax.transAxes)

        ax = axes[1, 0]
        r = ax.imshow(res[ifit][sel], origin='lower', cmap=gcmap)
        fig.colorbar(r, ax=ax)
        ax.text(0.1, 0.9, 'Residual', transform=ax.transAxes)

        p = pars[ifit]
        gax = axes[1, 1]
        gax = draw_ellipses(pars[ifit], gax, cmap=gcmap)
        pl.colorbar(dummy, ax=gax)

        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()
