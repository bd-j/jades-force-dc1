#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time, os, glob
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


def get_residuals(patcher, sceneDB, result_file):

    # --- get results from file ---
    with h5py.File(result_file, "r") as disk:
        active = disk["active"][:]
        fixed = disk["fixed"][:]
        chain = disk["chain"][:]
    config.seed_index = active["source_index"][0]

    # --- checkout region (parent operation) ---
    region, cactive, cfixed = sceneDB.checkout_region(seed_index=config.seed_index)
    if np.all(active["source_index"] == cactive["source_index"]):
        pass
    else:
        return None, None

    # --- Build patch on CPU side (child operation) ---
    patcher.build_patch(region, fixed, allbands=config.bandlist)
    pfixed = patcher.scene.get_proposal()
    original_data = np.split(patcher.data, np.cumsum(patcher.exposure_N)[:-1])

    # --- Send and subtract fixed sources ---
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()
    proposer = Proposer(patcher)
    out = proposer.evaluate_proposal(pfixed)
    fixed_residual = out[-1]

    # --- Send active sources, evaluate last element of chain ----
    patcher.pack_meta(active)
    patcher.swap_on_gpu()
    model = GPUPosterior(proposer, patcher.scene, verbose=False)
    model.evaluate(chain[-1, :])
    active_residual = model._residuals

    residuals = {"data": original_data,
                 "fixed_residual": fixed_residual,
                 "active_residual": active_residual,
                }
    extra = {"active_lnp": model._lnp,
             "active_grad": model._lnp_grad,
             "chain": chain,
             "active": active,
             "fixed": fixed
             }

    # patcher.free()  # this causes invalid device reference errors; is it necessary?
    return residuals, extra


if __name__ == "__main__":

    from config_mosaic import config
    parser.add_argument("--result_pattern", type=str, default="")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--no-reverse", dest="reverse", action="store_false")
    args = parser.parse_args()

    # --- combine cli arguments with config file arguments ---
    cargs = vars(config)
    cargs.update(vars(args))
    config = argparse.Namespace(**cargs)
    ingest_kwargs = {"rotate": config.rotate,
                     "reverse": config.reverse}

    # --- Build ingredients (parent and child sides) ---
    sceneDB = SuperScene(config.initial_catalog,
                         maxactive_per_patch=config.maxactive_per_patch,
                         ingest_kwargs=ingest_kwargs)
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)

    # --- files to get chains from ---
    result_list = glob.glob(os.path.expandvars(config.result_pattern))

    for result_file in result_list:
        print("working on ".format(result_file))
        residuals, extra = get_residuals(patcher, sceneDB, result_file)
        if residuals is None:
            print("could not generate scene for {}".format(result_file))
            continue
        fn = result_file.replace(".h5", "_mosaic_residuals.h5")
        dump_to_h5(fn, patcher, pixeldatadict=residuals,
                   otherdatadict=extra)
