#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior

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
    
    pass

def make_catalog(position, patch):
    """Inverse of JadesPatch.set_scene
    """
    nactive = patch.n_sources
    
    
    active = np.zeros(nactive, dtype=sourcecat_dtype)
    patch.scene.set_all_source_params(position)
    for i, row in enumerate(nactive):
        s = patch.scene.sources
        gid, x, y, q, pa, n, rh = s.ra, s.dec, s.q, s.pa, s.nsersic
        
        active[i]["nsersic"]
        active[i]["flux"][band_ids] = s.flux

            s.sersic = n
            s.rh = np.clip(rh, 0.05, 0.10)
            s.flux = flux[band_ids]
            s.ra = x
            s.dec = y
            s.q = np.clip(q, 0.2, 0.9)
            s.pa = np.deg2rad(pa)


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
    pfixed = patcher.scene.get_all_source_params().copy()

    # send patch to gpu, with space for residuals
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()

    # evaluate proposal for fixed sources
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pfixed)

    # make the pixel data pointer point to the residuals after subtracting fixed
    # sources; free the metadata pointers
    # patcher.switch_residual_pixel_pointer()  # TODO
    # patcher.free_meta()                      # TODO

    # --------------------------------
    # --- Run with active sources ----
    # --------------------------------

    # --- Update scene, re-pack and re-send meta ---
    # Update scene and pack new metadata
    patcher.pack_meta(activecat)
    pactive = patcher.scene.get_all_source_params().copy()

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

    # write chain

    # send back position and mass matrix


