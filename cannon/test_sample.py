#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
import numpy as np
import argparse
import h5py

import theano
import pymc3 as pm
import theano.tensor as tt

# child side
from forcepho.sources import Scene, Galaxy
from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior, LogLikeWithGrad
from jades_patch import JadesPatch
from mc import prior_bounds

# parent side
from catalog import rectify_catalog, catalog_to_scene
from dispatcher import SuperScene

# Local
#from catalog import rectify_catalog, catalog_to_scene, scene_to_catalog
from utils import Logger, dump_to_h5
parser = argparse.ArgumentParser()
theano.gof.compilelock.set_lock_status(False)


if __name__ == "__main__":

    verbose = True
    from config import config
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--no-reverse", dest="reverse", action="store_false")
    args = parser.parse_args()

    # --- combine cli arguments with config file arguments ---
    cargs = vars(config)
    cargs.update(vars(args))
    config = argparse.Namespace(**cargs)

    if config.logging:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
    else:
        logger = Logger(__name__)

    source_kwargs = {"splinedata": config.splinedatafile,
                     "free_sersic": True}
    logger.info("rotate is {}".format(config.rotate))
    logger.info("reverse is {}".format(config.reverse))

    # --- Build ingredients (parent and child sides) ---
    # sourcecat = rectify_catalog(config.initial_catalog, **ingest_kwargs)
    sourcecat, bands, header = rectify_catalog(config.initial_catalog,
                                               rotate=config.rotate,
                                               reverse=config.reverse)
    sceneDB = SuperScene(sourcecat=sourcecat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch)
    logger.info("Made SceneDB")
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile)
    logger.info("Made patch")
    active_scene = Scene()
    fixed_scene = Scene()

    # --- checkout region (parent operation) ---
    # seed_index = 444  # good source to build a scene from
    region, active, fixed = sceneDB.checkout_region(seed_index=config.seed_index)

    # This could be done within JadesPatch
    active_scene.from_catalog(active, bands, source_kwargs=source_kwargs)
    fixed_scene.from_catalog(fixed, bands, source_kwargs=source_kwargs)
    logger.info("checked out scene with {} active sources".format(len(active)))

    sr, sid = region.radius * 3600, active[0]["source_index"]
    ra, dec = region.ra, region.dec
    logger.info("scene of radius {:3.2f} arcsec centered on source "
                "{} at (ra, dec)=({}, {})".format(sr, sid, ra, dec))

    # --- Build patch on CPU side (child operation) ---
    # Note this is the *fixed* source metadata
    patcher.build_patch(region, scene=fixed_scene, allbands=config.bandlist)
    logger.info("built patch with {} fixed sources".format(len(fixed)))
    logger.info("Patch has {} pixels".format(len(patcher.data)))
    original = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])
    prop_fixed = patcher.scene.get_proposal()
    logger.info("got fixed proposal vector")

    # --- Send patch to GPU (with fixed sources) ---
    patcher.return_residual = True
    logger.info("Sending to gpu....")
    gpu_patch = patcher.send_to_gpu()
    logger.info("Initial Patch sent")

    # --- Evaluate (and subtract) fixed sources ---
    logger.info("Making proposer and sending fixed proposal")
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(prop_fixed)
    fixed_residual = out[-1]
    logger.info("Fixed sources subtracted")

    # --- Build active patch ---
    logger.info("Replacing cpu metadata with active sources")
    patcher.pack_meta(active_scene)
    p0 = patcher.scene.get_all_source_params().copy()
    logger.info("got active parameter vector")

    logger.info("Swapping fixed/active metadata and residual/data on GPU")
    patcher.swap_on_gpu()

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
            "fixed_residual": fixed_residual,
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
        fn = "patch{}_ra{:6.4f}_dec{:6.4f}.h5".format("sample", region.ra, region.dec)
    dump_to_h5(fn, proposer.patch, active, fixed,
               pixeldatadict=pixr, otherdatadict=extra)
    logger.info("wrote patch data to {}".format(fn))

    logger.info("Done")
