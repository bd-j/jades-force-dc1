#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob
import argparse
import numpy as np
from h5py import File


def make_psf_store(psfstorefile, mixture_directory,
                   nradii=9, data_dtype=np.float32):
    """Need to make an HDF5 file with <band>/psf datasets that have the form:
        psfs = np.zeros([nloc, nradii, ngauss], dtype=pdtype)
        pdtype = np.dtype([('gauss_params', np.float32, 6),
                           ('sersic_bin', np.int32)])
    and the order of `gauss_params` is given in patch.cu;
        amp, x, y, Cxx, Cyy, Cxy

    In this particular method we are using a single PSF for the entire
    image, and we are using the same number of gaussians (and same
    parameters) for each radius.

    Also, should store nradii (and maybe the radii it is meant to be used with.)
    Also, should store the total number of PSF gaussians per source in each band

    Parameters
    ------------

    psfstorefile : string
        The full path to the file where the PSF data will be stored.
        Must not exist.

    mixture_directory : dictionary
        This is a dictionary keyed by bands and with values that are a tuple of
        the path to the mixture file for that band and the number of the
        realization to use, e.g.:
            {"band": ("mixture_file", realization)}

    nradii : int (default, 9)
        The number of copies of the PSF to include, corresponding to the number
        of sersic mixture radii.  This is because in principle each of the
        sersic mixture radii can have a separate PSF mixture.

    data_type : np.dtype
        A datatype for the gaussian parameters
    """
    cols = ["amp", "xcen", "ycen", "Cxx", "Cyy", "Cxy"]
    dtype = [(c, data_dtype) for c in cols] + [("sersic_bin", np.int32)]
    psf_dtype = np.dtype(dtype)

    with File(psfstorefile, "x") as h5:
        for band, (mix, realization) in mixture_directory.items():
            bg = h5.create_group(band)

            with File(mix, "r") as pdat:
                allp = pdat["parameters"][realization]
            ngauss = len(allp)
            pars = np.zeros([1, nradii, ngauss], dtype=psf_dtype)
            bg.attrs["n_psf_per_source"] = int(nradii * ngauss)

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


def make_psf_directory(mixture_directory, search="gmpsf*ng4.h5"):
    psf_dir = {}
    mixes = glob.glob(os.path.join(mixture_directory, search))
    for mix in mixes:
        band = mix.split("_")[-2].upper()
        psf_dir[band] = (mix, 0)
    return psf_dir


if __name__ == "__main__":


    from argparse import Namespace
    config = Namespace()
    config.psfstorefile = "../stores/psf_test.h5"
    config.mixture_directory = "/Users/bjohnson/Projects/jades_force/data/psf/mixtures"
    config.psf_search = "gmpsf*ng4.h5"
    config.meta_dtype = np.float32

    # --- Make the PSF H5 file ---
    try:
        os.remove(config.psfstorefile)
    except:
        pass

    psf_dir = make_psf_directory(config.mixture_directory,
                                 config.psf_search)
    # can mess with the PSF directory here;
    # e.g., use ng=3 for some bands

    make_psf_store(config.psfstorefile, psf_dir,
                   data_dtype=config.meta_dtype,
                   nradii=9)
