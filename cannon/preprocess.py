#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import time

import numpy as np
import argparse
from h5py import File

from storage import ImageNameSet, PixelStore, MetaStore


def find_brants_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/br/",
                       pattern="udf_cube_*.slp.flat.fits"):
    search = os.path.join(os.path.expandvars(loc), pattern)
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f,                        # im
                          f.replace("slp", "err"),  # err
                          "",                       # mask
                          f.replace("slp", "bkg"))  # bkg
             for f in files]
    return names


def find_sandros_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/st/",
                        pattern="udf_cube_rev_*.flx.fits"):
    search = os.path.join(os.path.expandvars(loc), pattern)
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f,                        # im
                          f.replace("flx", "err"),  # err
                          f.replace("flx", "dia"),  # mask
                          f.replace("flx", "bkg"))  # bkg
             for f in files]
    return names


if __name__ == "__main__":

    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_directory", type=str,
                        default="$SCRATCH/eisenstein_lab/stacchella/mosaic/st")
    parser.add_argument("--store_directory", type=str,
                        default="$SCRATCH/eisenstein_lab/bdjohnson/jades_force/cannon/stores")
    parser.add_argument("--store_name", type=str,
                        default="mini-challenge-19-st")

    #from argparse import Namespace
    #config = Namespace()
    config = parser.parse_args()
    config.frames_directory = os.path.expandvars(config.frames_directory)
    config.store_directory = os.path.expandvars(config.store_directory)

    sd, sn = config.store_directory, config.store_name
    config.pixelstorefile = "{}/pixels_{}.h5".format(sd, sn)
    config.metastorefile = "{}/meta_{}.dat".format(sd, sn)
    config.nside_full = 2048
    config.super_pixel_size = 8
    config.pix_dtype = np.float32
    config.meta_dtype = np.float32
    config.bitmask = 1 | 256 | 1024 | 2048 | 4096 | 8192 | 16348

    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)
    # Make the (empty) metastore
    metastore = MetaStore()

    # --- Find Images ---
    names = find_sandros_images(loc=config.frames_directory)

    # Fill pixel and metastores
    for n in names:
        pixelstore.add_exposure(n, bitmask=config.bitmask)
        metastore.add_exposure(n)

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)

    print("done in {}s".format(time.time() - t))
