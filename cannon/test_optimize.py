#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
import numpy as np
import argparse
import h5py
from scipy.optimize import minimize

# child side
from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior, LogLikeWithGrad
from jades_patch import JadesPatch
from mc import prior_bounds

# parent side
from dispatcher import SuperScene

# Local
#from catalog import rectify_catalog, catalog_to_scene, scene_to_catalog
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

    # Build ingredients (parent and child sides)
    sceneDB = SuperScene(config.initial_catalog,
                         maxactive_per_patch=config.maxactive_per_patch,
                         ingest_kwargs=ingest_kwargs)
    sceneDB.sourcecat["nsersic"][:] = 2.0
    logger.info("Made SceneDB")
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)
    logger.info("Made patch")

    # checkout region (parent operation)
    # seed_index = 444  # good source to build a scene from
    region, active, fixed = sceneDB.checkout_region(seed_index=config.seed_index)
    logger.info("checked out scene with {} active sources".format(len(active)))
    sr, sid, ra, dec = region.radius*3600, active[0]["source_index"], region.ra, region.dec
    logger.info("scene of radius {:3.2f} arcsec centered on source {} at (ra, dec)=({}, {})".format(sr, sid, ra, dec))

    # Build patch on CPU side (child operation)
    # Note this is the *fixed* source metadata
    patcher.build_patch(region, fixed, allbands=config.bandlist)
    logger.info("built patch with {} fixed sources".format(len(fixed)))
    logger.info("Patch has {} pixels".format(len(patcher.data)))
    original = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])
    prop_fixed = patcher.scene.get_proposal()
    logger.info("got fixed proposal vector")

    # Send patch to GPU (with fixed sources)
    patcher.return_residual = True
    logger.info("Sending to gpu....")
    gpu_patch = patcher.send_to_gpu()
    logger.info("Initial Patch sent")

    # Evaluate (and subtract) fixed sources
    logger.info("Making proposer and sending fixed proposal")
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(prop_fixed)
    fixed_residual = out[-1]
    logger.info("Fixed sources subtracted")

    # Build active patch
    logger.info("Replacing cpu metadata with active sources")
    patcher.pack_meta(active)
    #print(patcher.scene)
    p0 = patcher.scene.get_all_source_params().copy()
    logger.info("got active parameter vector")

    logger.info("Swapping fixed/active metadata and residual/data on GPU")
    patcher.swap_on_gpu()

    # --- Instantiate the ln-likelihood object ---
    # This object reformats the Proposer return and splits the lnlike_function
    # into two, since that computes both lnp and lnp_grad, and we need to wrap
    # them in separate theano ops.
    verbose = True
    proposer = Proposer(patcher)
    model = GPUPosterior(proposer, patcher.scene, verbose=verbose)
    logger.info("Built posterior model")

    # run the pymc sampling
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)
    logger.info("Built loglike object")

    model.scene.set_all_source_params(p0)
    pnames = model.scene.parameter_names
    start = dict(zip(pnames, p0))

    if True:
        # --- Launch an optimization ---
        opts = {'ftol': 1e-6, 'gtol': 1e-6, 'factr': 10.,
                'disp':False, 'iprint': 1, 'maxcor': 20}
        theta0 = p0.copy()
        t = time.time()
        scires = minimize(model.nll, theta0, jac=True, method='BFGS',
                          options=opts, bounds=None)
        ts = time.time() - t
        chain = scires.x
        cols = ["fun", "nfev", "njev", "nit", "success", "status"]
        optinfo = np.zeros(1, dtype=np.dtype([(c, np.float) for c in cols]))
        for c in cols:
            v = getattr(scires, c)
            optinfo[c] = v
        print(scires.message)

    model.scene.set_all_source_params(chain)
    prop_last = model.scene.get_proposal()
    model.proposer.patch.return_residuals = True
    out = proposer.evaluate_proposal(prop_last)

    pixr = {"data": original,
            "fixed_residual": np.array(fixed_residual),
            "active_residual": np.array(out[-1]),
            }
    extra = {"active_chi2": out[0],
             "active_grad": out[1],
             "ncall": np.array(model.ncall),
             "chain": chain,
             "initial": p0,
             "opt": optinfo
             }

    fn = "patch{}_ra{:6.4f}_dec{:6.4f}.h5".format("optimize", region.ra, region.dec)
    dump_to_h5(fn, proposer.patch, active, fixed,
               pixeldatadict=pixr, otherdatadict=extra)
    logger.info("wrote patch data to {}".format(fn))

    logger.info("Done")