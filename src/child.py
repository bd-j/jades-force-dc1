#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior
from catalog import scene_to_catalog

logger = logging.getLogger(__name__)


def write_chain(region, patch, trace):
    
    # yuck.
    chain = np.array([trace.get_values(n) for n in pnames]).T
        
    result = Result()
    result.ndim = len(p0)
    result.nactive = miniscene.nactive
    result.nbands = patch.n_bands
    result.nexp = patch.n_exp
    result.pinitial = p0.copy()
    result.chain = chain
    result.ncall = np.copy(model.ncall)
    result.wall_time = ts
    #result.scene = miniscene
    result.lower = lower
    result.upper = upper
    result.patchname = patchname
    result.sky_reference = (pra, pdec)
    result.parameter_names = pnames

    return result


def run_patch(patcher, region, fixedcat, activecat, config):

    #isactive = sources["active"]
    #fixedcat = sources[~isactive]
    #activecat = sources[isactive]

    # ------------------------------
    # --- Subtract fixed sources ---
    # ------------------------------

    # Build the patch with fixed sources, packing pixel and meta-data arrays
    patcher.build_patch(region, fixedcat)
    # get the fixed source parameter vector
    pfixed = patcher.get_proposal()

    # send patch to gpu, with space for residuals
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()

    # evaluate proposal for fixed sources
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pfixed)

    # --------------------------------
    # --- Run with active sources ----
    # --------------------------------

    # --- Update scene, re-pack and re-send meta ---
    # Update scene and pack new metadata
    patcher.pack_meta(activecat)
    p0_active = patcher.get_proposal()

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
    trace, massout = run_hmc(model, p0_active, massin, config=config)

    # write chain
    write_chain(region, patcher, trace, config=config)
    niter = trace.

    # send back position and mass matrix
    cat = scene_to_catalog(patcher.scene)
    return cat, massout, fixed, niter

