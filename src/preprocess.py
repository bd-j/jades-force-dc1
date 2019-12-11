#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, sys

import numpy as np
import argparse
from h5py import File

from stores import ImageNameSet, PixelStore, MetaStore


def find_brants_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/br/"):
    search = loc + "udf_cube_*.slp.flat.fits"
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f, f.replace("slp", "err"), "", f.replace("slp", "bkg")) for f in files]
    return names


def find_sandros_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/mosaics/st/"):
    search = loc + "udf_cube_*.slp.fits"
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f, f.replace("slp", "err"), "", f.replace("slp", "bkg")) for f in files]
    return names


def flux_calibrate():
    pass




if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #config = parser.parse_args()
    from argparse import Namespace
    config = Namespace()
    config.pixelstorefile = "stores/pixels_test.h5"
    config.metastorefile = "stores/meta_test.dat"
    config.nside_full = 2048
    config.super_pixel_size = 8
    config.pix_dtype = np.float32
    config.meta_dtype = np.float32
    #config.psfstorefile = "stores/psf_test.h5"
    #config.mixture_directory = "/Users/bjohnson/Projects/jades_force/data/psf/mixtures"
    #config.psf_search = "gmpsf*ng4.h5"

    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)
    # Make the (empty) metastore
    metastore = MetaStore()

    # --- Find Images ---
    names = find_brants_images()

    # Fill pixel and metastores
    for n in names[:1]:
        pixelstore.add_exposure(n)
        metastore.add_exposure(n)

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)
