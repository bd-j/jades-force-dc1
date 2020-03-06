#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits

expandpath = os.path.expandvars
pjoin = os.path.join

JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]

prop_cycle = pl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def flux_matrix(cat, objid=slice(None), bands=JWST_BANDS):
    """
    Returns
    -------
    F : ndarray, shape (nobj, nband, nsample)
    """
    fluxes = np.array([cat[objid][b] for b in bands])
    return np.squeeze(fluxes.transpose(1, 0, 2))


def get_color_chain(chaincat, i, j, point=False):
    fr = chaincat[JWST_BANDS[i]] / chaincat[JWST_BANDS[j]]
    color = -2.5 * np.log10(fr)
    if point:
        return np.mean(color, axis=-1), np.std(color, axis=-1)
    else:
        return color, None


def get_color_sandro(cat, i, j):
    fr = sandro["flux"][:, i] / sandro["flux"][:, j]
    u1 = 1.086 * sandro["flux_unc"][:, i] / sandro["flux"][:, i]
    u2 = 1.086 * sandro["flux_unc"][:, j] / sandro["flux"][:, j]
    return -2.5*np.log10(fr), np.hypot(u1, u2)


def show_chain(cat, ind, bins=20, range=None, hist=True):
    fig, axes = pl.subplots(len(JWST_BANDS))
    if hist:
        for i, ax in enumerate(axes):
           ax.hist(cat[ind][JWST_BANDS[i]], bins=bins, range=range)
           ax.set_xlabel(JWST_BANDS[i])
    else:
        for i, ax in enumerate(axes):
            ax.plot(cat[ind][JWST_BANDS[i]])
            ax.set_ylabel(JWST_BANDS[i])

    return fig, axes


if __name__ == "__main__":

    sandro_file = expandpath("$HOME/Projects/jades_force/data/2019-mini-challenge/source_catalogs/forcepho_table_psf_matched_v5.0.fits")
    chaincat_file = expandpath("$HOME/Projects/jades_force/cannon/chains_mini-challenge-19_v0.fits")
    summary_file = expandpath("$HOME/Projects/jades_force/cannon/summary_mini-challenge-19_v0.fits")

    sandro = fits.getdata(sandro_file)
    chaincat = fits.getdata(chaincat_file)
    summary = fits.getdata(summary_file)

    inds = summary["id"]
    sandro = sandro[inds]

    F = flux_matrix(chaincat)
    C = [np.cov(f) for f in F]
    xx = np.linspace(0, 100, 100)

    # flux ratios
    fr = np.array([summary[JWST_BANDS[ib]] / sandro["flux"][:, ib]
                   for ib in range(9)]).T
    snr_sandro = sandro["flux"] / sandro["flux_unc"]
    snr_force = np.array([summary[JWST_BANDS[ib]] / summary["{}_unc".format(JWST_BANDS[ib])]
                          for ib in range(9)]).T
    frunc = np.hypot(fr / snr_sandro, fr / snr_force)

    # outliers
    bbl = (sandro["flux"] > 75) & (fr < 0.8)
    badlow = np.sum(bbl, axis=-1) > 0

    bbh = (sandro["flux"] > 20) & (fr > 2.5)
    badhigh = np.sum(bbh, axis=-1) > 0

    if False:
        inds = np.random.choice(len(C), size=(9,)).astype(int)
        fig, axes = pl.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(C[inds[i]].T, origin="lower")
            ax.set_title("source {}".format(chaincat[inds[i]]["id"]+ 1))

    if False:
        fig, ax = pl.subplots()
        # Flux comparison
        ax.errorbar(sandro["flux"][:, ib], summary[JWST_BANDS[ib]],
                    xerr=sandro["flux_unc"][:, ib],
                    yerr=summary["{}_unc".format(JWST_BANDS[ib])],
                    marker="o", linestyle="", label=JWST_BANDS[ib])
        ax.set_xlabel("Flux (Sandro, nJy)")
        ax.set_ylabel("Flux (force, nJy)")
        ax.plot(xx, xx, linestyle=":", color="k")

    # Flux error comparison
    if False:
        fig, ax = pl.subplots()
        # err vs err
        ax.plot(sandro[inds]["flux_unc"][:, ib], summary["{}_unc".format(JWST_BANDS[ib])], 'o')
        # err vs flux for both
        ax.plot(summary[JWST_BANDS[ib]], summary["{}_unc".format(JWST_BANDS[ib])], 'o')
        ax.plot(sandro["flux"][:, ib], sandro["flux_unc"][:, ib], 'o')
        # snr vs snr
        for ib in [0, 3, 8]:
            ax.plot(snr_sandro[:, ib], snr_force[:, ib],
                    marker='o', markersize=3, legend=JWST_BANDS[ib])
        ax.plot(xx, xx*4, linestyle=":", color="k")
        ax.set_xlabel("SNR (Sandro)")
        ax.set_ylabel("SNR (force)")


    # plot flux ratio
    if True:
        fig, ax = pl.subplots(figsize=(14, 5.))
        for ib in [0, 3, 6, 8]:
            ax.errorbar(sandro["flux"][:, ib], fr[:, ib], yerr=frunc[:, ib],
                        marker="", linestyle="", alpha=0.5, color=colors[ib])
            ax.plot(sandro["flux"][:, ib], fr[:, ib], marker="o", markersize=3,
                    label=JWST_BANDS[ib], color=colors[ib], linestyle="")
        ax.plot(xx * 4, np.ones_like(xx), linestyle=":", color="k")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(-1, 8)
        ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("Flux (SEP, nJy)")
        ax.set_ylabel("Flux ratio (force /SEP)")
        fig.savefig("fluxratio_2.png", dpi=450)

    if True:
        cp = [(0, 1), (1, 3), (3, -1)]
        fig, axes = pl.subplots(1, len(cp), figsize=(14, 5.))

        for k, (i, j) in enumerate(cp):
            cnames = JWST_BANDS[i], JWST_BANDS[j]
            ax = axes.flat[k]
            fc, fce = get_color_chain(chaincat, i, j, point=True)
            sc, sce = get_color_sandro(sandro, i, j)

            sel = snr_sandro[:, 3] > 10
            ax.errorbar(sc[sel], fc[sel], xerr=sce[sel], yerr=fce[sel],
                        alpha=0.5, marker="", linestyle="", color=colors[k])
            ax.plot(sc[sel], fc[sel], marker="o", markersize=3,
                        linestyle="", color=colors[k], label="{} - {}".format(*cnames))

        [ax.set_xlim(-2, 2.5) for ax in axes.flat]
        [ax.set_ylim(-2, 2.5) for ax in axes.flat]
        [ax.legend() for ax in axes.flat]
        [ax.set_xlabel("Color (SEP, mags)") for ax in axes.flat]
        axes[0].set_ylabel("Color (force, mags)")
        [ax.plot(xx[:5]-2, xx[:5]-2, color="k", linestyle="--") for ax in axes.flat]
        [ax.plot(xx[:5]-2, xx[:5]-2 + 0.2, color="k", linestyle=":") for ax in axes.flat]
        [ax.plot(xx[:5]-2, xx[:5]-2 - 0.2, color="k", linestyle=":") for ax in axes.flat]
        fig.savefig("color_f200wsnr=10.png", dpi=450)

    # Positional comparison
    if False:
        fig, ax = pl.subplots()
        ra2pix = 3600 * np.cos(np.deg2rad(summary["dec"])) / 0.03
        dx = (sandro["ra"] - summary["ra"]) * ra2pix
        dy = (sandro["dec"] - summary["dec"]) * 3600 / 0.03
        edy = summary["ra_unc"] * ra2pix
        edx = summary["dec_unc"] * 3600 / 0.03

        ax.errorbar(dx, dy, xerr=edx, yerr=edy, marker='.', linestyle="")
        ax.set_xlabel(r"$\Delta$ RA (SW pixels)")
        ax.set_ylabel(r"$\Delta$ DEC (SW pixels)")
