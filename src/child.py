#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import logging

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior, LogLikeWithGrad
from forcepho.fitting import Result
from catalog import scene_to_catalog
from mc import prior_bounds

logger = logging.getLogger(__name__)


def run_patch(patcher, region, fixedcat, activecat,
              config, logger=logger):

    #isactive = sources["active"]
    #fixedcat = sources[~isactive]
    #activecat = sources[isactive]

    # ------------------------------
    # --- Subtract fixed sources ---
    # ------------------------------

    # --- Build the patch with fixed sources ---
    # packs pixels and meta data 
    patcher.build_patch(region, fixedcat, allbands=config.bandlist)
    logger.info("built patch with {} fixed sources".format(len(fixed)))
    logger.info("Patch has {} pixels".format(len(patcher.data)))
    original = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])

    # --- Send patch to GPU (with fixed sources) ---
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()
    logger.info("Initial fixed Patch sent to GPU")

    # --- Evaluate (and subtract) fixed sources ---
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(prop_fixed)
    fixed_residual = out[-1]
    logger.info("Fixed sources subtracted")

    # --------------------------------
    # --- Run with active sources ----
    # --------------------------------

    # --- Update scene, re-pack and re-send meta ---
    # Update scene, pack new metadata, send to GPU, swap data and residual
    patcher.pack_meta(activecat)
    patcher.swap_on_gpu()
    logger.info("Swapped fixed/active metadata and residual/data on GPU")
    # here we get the scene parameter vector (used for HMC),
    # not the GPU proposal vector
    p0 = patcher.scene.get_all_source_params().copy()
    pnames = patcher.scene.parameter_names

    # --- Instantiate the ln-likelihood object ---
    # This object reformats the Proposer return and splits the lnlike_function
    # into two, since that computes both lnp and lnp_grad, and we need to wrap
    # them in separate theano ops.
    model = GPUPosterior(proposer, patcher.scene, verbose=config.verbose)
    lnp0 = model.lnprob(p0)
    logger.info("Initial lnp={}".format(lnp0))

    # launch HMC
    tstart = time.time()
    trace, step = run_pymc3(model, p0, config, init_mass=massin)
    twall = time.time() - tstart

    # get things to save
    chain = trace.get_values("proposal")
    model.proposer.patch.return_residuals = True
    model.evaluate(chain[-1, :])
    active_residual = model._residuals
    cat = patch_to_catalog(patcher)

    # write info for this run
    result = make_result(region, patcher, trace, residuals, **extras)
    result.dump_to_h5(filename)

    # send back position and mass matrix
    return cat, massout, fixed, niter


def run_pymc3(model, config):
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)
    with pm.Model() as opmodel:
        # set priors for each element of theta
        z0, start = prior_bounds(model.scene)
        theta = tt.as_tensor_variable(z0[0])
        # instantiate target density and start sampling.
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=config.n_iter,
                          tune=config.n_warm,
                          start=start,
                          compute_convergence_checks=False,
                          cores=1, progressbar=config.show_progress,
                          discard_tuned_samples=True)
    return trace, None




def make_result(region, patch, trace, ncall=-1, twall=0, config=None):

    # yuck.
    pnames = patch.scene.parameter_names
    chain = np.array([trace.get_values(n) for n in pnames]).T

    result = Result()
    result.chain = chain
    result.niter = chain.shape[0]

    # inputs
    result.active = active
    result.fixed = fixed
    result.nbands = patch.bandlist
    result.exposures = patch.epaths
    result.priors = False # FIXME

    # meta info
    result.ncall = ncall
    result.wall_time = twall
    result.patchname = patchname

    result.patch_reference_coordinates = patch.patch_reference_coordinates
    result.parameter_names = pnames
    result.niter = chain.shape[0]

    return result

