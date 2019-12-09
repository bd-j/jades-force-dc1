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


def make_psf_store(psfstorefile, mixture_directory, search="gmpsf*ng4.h5",
                   psf_realization=0, nradii=9, data_dtype=np.float32):
    """Need to make band/psf datasets with:
        psfs = np.zeros([nloc, nradii, ngauss], dtype=pdt)
        pdt = np.dtype([('gauss_params', np.float32, 6),
                        ('sersic_bin', np.int32)])
    and the order of gauss_params is given in patch.cu;
        amp, x, y, Cxx, Cyy, Cxy

    In this method we are suing a single PSF for the entire image, and we
    are using the same number of gaussians (and same parameters) for each
    radius.

    This should really take a dictionary of {"band": ("mixture_file", realization) for more flexibility

    """
    cols = ["amp", "xcen", "ycen", "Cxx", "Cyy", "Cxy"]
    dtype = [(c, data_dtype) for c in cols] + [("sersic_bin", np.int32)]
    psf_dtype = np.dtype(dtype)

    mixes = glob.glob(os.path.join(mixture_directory, search))
    with File(psfstorefile, "x") as h5:
        for mix in mixes:
            band = mix.split("_")[-2].upper()
            bg = h5.create_group(band)

            with File(mix, "r") as pdat:
                allp = pdat["parameters"][psf_realization]
            ngauss = len(allp)
            pars = np.zeros([1, nradii, ngauss], dtype=psf_dtype)

            # Fill every radius with the parameters for the ngauss gaussians
            pars["amp"]  = allp["amp"]
            pars["xcen"] = allp["x"]
            pars["ycen"] = allp["y"]
            pars["Cxx"]  = allp["vxx"]
            pars["Cyy"]  = allp["vyy"]
            pars["Cxy"]  = allp["vxy"]
            pars["sersic_bin"] = np.arange(nradii)[None, :, None]
            bg.create_dataset("parameters", data=pars.reshape(1, -1))
            #bg.create_dataset("detector_locations", data=pixel_grid)


if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #config = parser.parse_args()
    from argparse import Namespace
    config = Namespace()
    config.psfstorefile = "stores/psf_test.h5"
    config.pixelstorefile = "stores/pixels_test.h5"
    config.metastorefile = "stores/meta_test.dat"
    config.nside_full = 2048
    config.super_pixel_size = 8
    config.mixture_directory = "/Users/bjohnson/Projects/jades_force/data/psf/mixtures"
    config.psf_search = "gmpsf*ng4.h5"
    config.pix_dtype = np.float32
    config.meta_dtype = np.float32


    # --- Make the PSF H5 file ---
    try:
        os.remove(config.psfstorefile)
    except:
        pass
    make_psf_store(config.psfstorefile,
                   config.mixture_directory,
                   search=config.psf_search,
                   data_dtype=config.meta_dtype,
                   nradii=9, psf_realization=0)

    sys.exit()
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
