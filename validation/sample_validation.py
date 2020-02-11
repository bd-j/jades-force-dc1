#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
from astropy.io import fits

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior, LogLikeWithGrad
from mc import prior_bounds

from jades_patch import JadesPatch
from region import RectangularRegion
from utils import Logger, dump_to_h5

import pymc3 as pm


def checkout_region(catalog, i, j, buffer=20*0.03/3600.):
    """Assumes source catalog consists of 256 sources in order, with ra
    coordinate changing fastest. Output is a rectangular region and a set of 16
    active sources arranged 4x4 in sky coordinates.
    """
    ind = np.arange(len(catalog))

    x = np.mod(ind, 16)
    y = np.floor(ind / 16).astype(int)
    sx = np.floor(x / 4).astype(int)
    sy = np.floor(y / 4).astype(int)

    k = (sx == i) & (sy == j)
    acat = catalog[k]

    ra_min = acat["ra"].min() - buffer / np.cos(np.deg2rad(acat["dec"].mean()))
    ra_max = acat["ra"].max() + buffer / np.cos(np.deg2rad(acat["dec"].mean()))
    dec_min = acat["dec"].min() - buffer
    dec_max = acat["dec"].max() + buffer

    region = RectangularRegion(ra_min, ra_max, dec_min, dec_max)

    return region, acat


if __name__ == "__main__":

    from config_validation import config
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_num", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    # --- combine cli arguments with config file arguments ---
    cargs = vars(config)
    cargs.update(vars(args))
    config = argparse.Namespace(**cargs)

    patch_i = np.mod(config.patch_num, 4)
    patch_j = np.floor(config.patch_num / 4).astype(int)

    if config.logging:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
    else:
        logger = Logger(__name__)

    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)

    # --- checkout region (parent operation) ---
    catalog = fits.getdata(config.initial_catalog)
    region, active = checkout_region(catalog, patch_i, patch_j)

    # --- Build patch on CPU side ---
    patcher.build_patch(region, active, allbands=config.bandlist)
    original = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])
    logger.info("built patch on CPU")

    if config.display:
        import matplotlib.pyplot as pl
        from show_patch import split_patch_exp, show_exp, sky_to_pix
        x, y, d, i = split_patch_exp(patcher)
        fig, ax = pl.subplots()
        i = 0
        ax = show_exp(x[i], y[i], d[i], ax=ax)
        pix = sky_to_pix(active["ra"], active["dec"], patch=patcher, exp_idx=i)
        ax.plot(pix[:, 0], pix[:, 1], marker='x', color='red', linestyle='')
        sys.exit()

    # --- Send patch to GPU ---
    patcher.return_residual = True
    logger.info("Sending to gpu....")
    gpu_patch = patcher.send_to_gpu()
    logger.info("Initial Patch sent")

    # --- Build active patch ---
    p0 = patcher.scene.get_all_source_params().copy()
    logger.info("got active parameter vector")

    # --- Instantiate the ln-likelihood object ---
    # This object reformats the Proposer return and splits the lnlike_function
    # into two, since that computes both lnp and lnp_grad, and we need to wrap
    # them in separate theano ops.
    proposer = Proposer(patcher)
    model = GPUPosterior(proposer, patcher.scene, verbose=verbose)
    logger.info("Built posterior model")
    lnp0 = model.lnprob(p0)
    logger.info("Initial lnp={}".format(lnp0))
    pnames = model.scene.parameter_names

    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)
    logger.info("Built loglike object")

    # --- Run hmc (simple) ---
    logger.info("Begin sampling with {} warm and "
                "{} iterations".format(config.n_warm, config.n_iter))

    with pm.Model() as opmodel:
        # set priors for each element of theta
        z0, start = prior_bounds(model.scene)
        logger.info("got priors")
        theta = tt.as_tensor_variable(z0[0])
        # instantiate target density and start sampling.
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=config.n_iter,
                          tune=config.n_warm,
                          start=start,
                          compute_convergence_checks=False,
                          cores=1, progressbar=config.show_progress,
                          discard_tuned_samples=True)
    logger.info("Done sampling")
    chain = trace.get_values("proposal")

    # Failsafes
    from astropy.io import fits
    fits.writeto("stest_chain_id{}.fits".format(active[0]["source_index"]), chain)
    logger.info("Got {} samples.".format(chain.shape[0]))
    logger.info("Last position is: \n {}".format(chain[-1, :]))

    # now store residuals and other important info
    model.scene.set_all_source_params(chain[-1, :])
    prop_last = model.scene.get_proposal()
    model.proposer.patch.return_residuals = True
    out = proposer.evaluate_proposal(prop_last)

    pixr = {"data": original,
            "active_residual": out[-1],
            }
    extra = {"active_chi2": out[0],
             "active_grad": out[1],
             "ncall": model.ncall,
             "chain": chain,
             "reference_coordinates": patcher.patch_reference_coordinates,
             "region": np.array([region.ra, region.dec, region.radius])
             }

    if config.outfile:
        fn = config.outfile
    else:
        fn = "validation_{}.h5".format(patchid)
    dump_to_h5(fn, proposer.patch, active, fixed,
               pixeldatadict=pixr, otherdatadict=extra)
    logger.info("wrote patch data to {}".format(fn))
    logger.info("Done")