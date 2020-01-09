#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
#import matplotlib.pyplot as pl
import logging

# child side
from jades_patch import JadesPatch
from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior

# parent side
from dispatcher import SuperScene

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    from config import config
    logger = logging.getLogger(__name__)

    # Build ingredients (parent and child sides)
    sceneDB = SuperScene(config.initial_catalog)
    logger.info("Made SceneDB")
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)
    logger.info("Made patch")

    # checkout region (parent operation)
    # seed_index = 444  # good source to build a scene from
    region, active, fixed = sceneDB.checkout_region()
    logger.info("checked out scene")

    # Build patch on CPU side (child operation)
    patcher.build_patch(region, fixed, allbands=config.bandlist)
    logger.info("built patch")
    pfixed = patcher.scene.get_all_source_params().copy()
    logger.info("got proposal vector")
    print(pfixed)

    # Send patch to GPU
    patcher.return_residual = True
    logger.info("Sending to gpu....")
    gpu_patch = patcher.send_to_gpu()
    logger.info("Patch sent")

    # Build active patch
    logger.info("Replacing cpu metadata with active sources")
    patcher.pack_meta(active)
    #print(patcher.scene)
    pactive = patcher.scene.get_all_source_params().copy()

    #sys.exit()

    logger.info("Swapping metadata and residual/data on GPU")
    patcher.swap_on_gpu()

    # send proposal to GPU
    logger.info("Making proposer and sending proposal")
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pactive)
