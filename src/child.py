#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior
from forcepho.fitting import Result
from catalog import scene_to_catalog

logger = logging.getLogger(__name__)


def run_patch(patcher, region, fixedcat, activecat, config):

    #isactive = sources["active"]
    #fixedcat = sources[~isactive]
    #activecat = sources[isactive]

    # ------------------------------
    # --- Subtract fixed sources ---
    # ------------------------------

    # Build the patch with fixed sources, packing pixel and meta-data arrays
    patcher.build_patch(region, fixedcat)

    # send patch to gpu, with space for residuals
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()

    # evaluate proposal for fixed sources
    prop_fixed = patcher.get_proposal()
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(prop_fixed)

    # --------------------------------
    # --- Run with active sources ----
    # --------------------------------

    # --- Update scene, re-pack and re-send meta ---
    # Update scene and pack new metadata
    patcher.pack_meta(activecat)
    # here we need the scene parameter vector (used for HMC),
    # not the actual GPU proposal vector
    p0 = patcher.scene.get_all_source_params().copy()

    # This method:
    # 1) frees exisiting metadata arrays on GPU;
    # 2) sends new metadata arrays to device, updating CUDA pointers;
    # 3) swaps CUDA pointers for the device-side residual and data arrays;
    # 4) deletes existing patch structure on device;
    # 5) Creates and fills new patch structure with new pointers and sends to device;
    patcher.swap_on_gpu()

    # --- Instantiate the ln-likelihood object ---
    # This object reformats the Proposer return and splits the lnlike_function
    # into two, since that computes both lnp and lnp_grad, and we need to wrap
    # them in separate theano ops.
    model = GPUPosterior(proposer, patcher.scene, verbose=verbose)

    # launch HMC
    tstart = time.time()
    trace, step = run_pymc3(model, p0, config, init_mass=massin)
    twall = time.time() - tstart

    # write chain
    result = make_result(region, patcher, trace,
                         ncall=model.ncall.copy(), config=config)
    result.dump_to_h5(filename)

    # send back position and mass matrix
    cat = scene_to_catalog(patcher.scene)
    return cat, massout, fixed, niter


def run_pymc3(model, p0, config, init_mass=None):

    # -- Launch HMC ---
    # wrap the loglike and grad in theano tensor ops
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)

    # Get upper and lower bounds for variables
    lower, upper = prior_bounds(miniscene)
    print(lower.dtype, upper.dtype)
    pnames = model.scene.parameter_names
    start = dict(zip(pnames, p0))

    # Define the windows used to tune the mass matrix.
    k = np.floor(np.log2((config.n_tune - config.n_warm) / config.n_start))
    nwindow = config.n_start * 2 ** np.arange(k)
    nwindow = np.append(nwindow, config.n_tune - config.n_warm - np.sum(nwindow))
    nwindow = nwindow.astype(int)

    # The pm.sample() method below will draw an initial theta,
    # then call logl.perform and logl.grad multiple times
    # in a loop with different theta values.
    with pm.Model() as opmodel:

        # Set priors for each element of theta.
        z0 = prior_bounds(model.scene)
        theta = tt.as_tensor_variable(z0)

        # Instantiate target density.
        pm.DensityDist('likelihood', lambda v: logl(v),
                        observed={'v': theta})

        # Tune mass matrix.
        start = None
        burnin = None
        for steps in nwindow:
            step = get_step_for_trace(init_cov=init_cov, trace=burnin)
            burnin = pm.sample(start=start, tune=steps, draws=2, step=step,
                               compute_convergence_checks=False,
                               discard_tuned_samples=False)
            start = [t[-1] for t in burnin._straces.values()]
        step = get_step_for_trace(init_cov=init_cov, trace=burnin)

        # Sample with tuned mass matrix.
        trace = pm.sample(draws=config.n_iter, tune=config.n_warm,
                          step=step, start=start,
                          progressbar=False, cores=1,
                          discard_tuned_samples=True)

        return trace, step


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

