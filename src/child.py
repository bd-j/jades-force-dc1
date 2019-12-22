#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior

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


