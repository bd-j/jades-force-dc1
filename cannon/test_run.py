#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
import numpy as np
import argparse
import h5py

# child side
from jades_patch import JadesPatch
from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior

# parent side
from dispatcher import SuperScene

# Local
#from catalog import rectify_catalog
from utils import Logger, dump_to_h5
parser = argparse.ArgumentParser()


if __name__ == "__main__":

    from config import config
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--ntime", type=int, default=0)
    parser.add_argument("--check_grad", action="store_true")
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

    ingest_kwargs = {"rotate": config.rotate,
                     "reverse": config.reverse}

    logger.info("rotate is {}".format(config.rotate))
    logger.info("reverse is {}".format(config.reverse))

    # --- Build ingredients (parent and child sides) ---
    sceneDB = SuperScene(config.initial_catalog,
                         maxactive_per_patch=config.maxactive_per_patch,
                         ingest_kwargs=ingest_kwargs)
    logger.info("Made SceneDB")
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)
    logger.info("Made patch")

    # --- checkout region (parent operation) ---
    # seed_index = 444  # good source to build a scene from
    region, active, fixed = sceneDB.checkout_region(seed_index=config.seed_index)
    logger.info("checked out scene with {} active sources".format(len(active)))
    sr, sid, ra, dec = region.radius*3600, active[0]["source_index"], region.ra, region.dec
    logger.info("scene of radius {:3.2f} arcsec centered on source {} at (ra, dec)=({}, {})".format(sr, sid, ra, dec))

    # --- Build patch on CPU side (child operation) ---
    # Note this is the *fixed* source metadata
    patcher.build_patch(region, fixed, allbands=config.bandlist)
    logger.info("built patch with {} fixed sources".format(len(fixed)))
    logger.info("Patch has {} pixels".format(len(patcher.data)))
    original = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])
    parfixed = patcher.scene.get_all_source_params().copy()
    pfixed = patcher.scene.get_proposal()
    logger.info("got fixed proposal vector")

    # --- Send patch to GPU (with fixed sources) ---
    patcher.return_residual = True
    logger.info("Sending to gpu....")
    gpu_patch = patcher.send_to_gpu()
    logger.info("Initial Patch sent")

    # --- Evaluate (and subtract) fixed sources ---
    logger.info("Making proposer and sending fixed proposal")
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pfixed)
    fixed_residual = out[-1]
    logger.info("Fixed sources subtracted")

    # --- Build active patch ----
    logger.info("Replacing cpu metadata with active sources")
    patcher.pack_meta(active)
    paractive = patcher.scene.get_all_source_params().copy()
    pactive = patcher.scene.get_proposal()
    logger.info("got active proposal vector")

    logger.info("Swapping fixed/active metadata and residual/data on GPU")
    patcher.swap_on_gpu()

    # --- send proposal to GPU ---
    #patcher.return_residual = False
    logger.info("Making new proposer and sending active proposal")
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pactive)

    # --- Write patch info and residuals ---
    pixr = {"data": original,
            "fixed_residual": np.array(fixed_residual),
            "active_residual": np.array(out[-1]),
            }
    extra = {"active_chi2": out[0],
             "active_grad": out[1]
             }

    if config.outfile:
        fn = config.outfile
    else:
        fn = "patch{}_ra{:6.4f}_dec{:6.4f}.h5".format("test", region.ra, region.dec)
    dump_to_h5(fn, proposer.patch, active, fixed,
               pixeldatadict=pixr, otherdatadict=extra)
    logger.info("wrote patch data to {}".format(fn))

    # --- Do some checks ---

    if config.check_grad:
        proposer.patch.return_residual = False
        z0 = paractive.copy()
        model = GPUPosterior(proposer, patcher.scene, verbose=True)
        lnp = model.lnprob(z0)
        delta_lnp = 0.1
        delta = delta_lnp / dlnp

        # dlnp, dlnp_num = model.check_grad(z0, delta)
        dlnp = model.lnprob_grad(z0)
        dlnp_num = np.zeros(len(z0), dtype=np.float64)
        for i, dp in enumerate(delta):
            theta = z0.copy()
            theta[i] -= dp
            model.evaluate(theta)
            imlo = model.lnprob(theta)
            theta[i] += 2 * dp
            model.evaluate(theta)
            imhi = model.lnprob(theta)
            dlnp_num[i] = ((imhi - imlo) / (2 * dp))

    if config.ntime > 0:
        logger.info("Timing proposal evaluation")
        proposer.patch.return_residual = False
        tstart = time.time()
        for i in range(config.ntime):
            o = proposer.evaluate_proposal(pactive)
        dur = time.time() - tstart
        logger.info("Completed {} proposals in {}s".format(ntime, dur))

    logger.info("Done")