#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import time

import numpy as np
import argparse
from h5py import File
from astropy.io import fits

from storage import ImageNameSet, PixelStore, MetaStore


def find_images(loc=".", pattern="*sci.fits"):
    import glob
    search = os.path.join(config.frames_directory, pattern)
    files = glob.glob(search)
    names = [ImageNameSet(f,                        # im
                          f.replace("sci", "unc"),  # err
                          "",  # mask
                          "")  # bkg
             for f in files]
    return names


def rectify_images(scinames, **header_kwargs):
    """Add a FILTER keyword to images
    """
    for n in scinames:
        with fits.open(n, mode="update") as hdul:
            hdul[0].header.update(**header_kwargs)
            hdr = hdul[0].header
            dat = hdul[0].data
        unc = np.ones_like(dat)
        fits.writeto(n.replace("sci", "unc"), unc, header=hdr, overwrite=True)


def make_psf_store(psfstorefile, nradii=9, band="", 
                   fwhm=[3.0], amp=[1.0],
                   data_dtype=np.float32):
    """Need to make an HDF5 file with <band>/psf datasets that have the form:
        psfs = np.zeros([nloc, nradii, ngauss], dtype=pdtype)
        pdtype = np.dtype([('gauss_params', np.float32, 6),
                           ('sersic_bin', np.int32)])
    and the order of `gauss_params` is given in patch.cu;
        amp, x, y, Cxx, Cyy, Cxy

    In this particular method we are using a single PSF for the entire
    image, and we are using the same number of gaussians (and same
    parameters) for each radius.

    Parameters
    ------------
    psfstorefile : string
        The full path to the file where the PSF data will be stored.
        Must not exist.

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

    with File(psfstorefile, "a") as h5:
        try:
            bg = h5.create_group(band)
        except(ValueError):
            del h5[band]
            bg = h5.create_group(band)

        ngauss = len(fwhm)
        amps = np.ones(ngauss) * np.array(amp)
        amps /= amps.sum()
        sigma = np.array(fwhm) / 2.35
        pars = np.zeros([1, nradii, ngauss], dtype=psf_dtype)
        bg.attrs["n_psf_per_source"] = int(nradii * ngauss)

        # Fill every radius with the parameters for the ngauss gaussians
        pars["amp"]     = amps
        pars["xcen"][:] = 0.0
        pars["ycen"][:] = 0.0
        pars["Cxx"]     = sigma**2
        pars["Cyy"]     = sigma**2
        pars["Cxy"]     = 0.0
        pars["sersic_bin"] = np.arange(nradii)[None, :, None]
        bg.create_dataset("parameters", data=pars.reshape(1, -1))
        #bg.create_dataset("detector_locations", data=pixel_grid)


def rectify_catalog(catfile, outcatname, imname):
    from astropy.wcs import WCS

    x, y = np.genfromtxt(catfile, skip_header=1).T
    hdr = fits.getheader(imname)
    wcs = WCS(hdr)
    ra, dec = wcs.all_pix2world(x, y, 1)

    bands = [hdr["FILTER"]]
    from catalog import sourcecat_dtype
    dt = sourcecat_dtype(bands=bands)

    sourcecat = np.zeros(len(ra), dtype=dt)
    sourcecat["ra"] = ra
    sourcecat["dec"] = dec
    sourcecat["flux"][:] = 10.0
    sourcecat["rhalf"] = 0.1
    sourcecat["nsersic"] = 3.0
    sourcecat["q"] = 0.85
    sourcecat["source_index"][:] = np.arange(len(ra))

    #write the fits file
    fits.writeto(outcatname, sourcecat, overwrite=True)
    with fits.open(outcatname, mode="update") as hdul:
        hdul[0].header["FILTERS"] = ",".join(bands)


if __name__ == "__main__":

    # --- Configuration ---
    # ---------------------

    # read config file
    from config_validation import config

    # read command lines
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_directory", type=str,
                        default="./stores")
    parser.add_argument("--version", type=str,
                        default="v0")
    parser.add_argument("--rectify", action="store_true")
    args = parser.parse_args()

    # Combine cli arguments with config file arguments ---
    cargs = vars(config)
    cargs.update(vars(args))
    config = argparse.Namespace(**cargs)

    # --- initialize some names ----
    config.store_name = "galsim_{}".format(config.version)
    sd, sn = config.store_directory, config.store_name
    config.pixelstorefile = "{}/pixels_{}.h5".format(sd, sn)
    config.metastorefile = "{}/meta_{}.dat".format(sd, sn)
    config.psfstorefile = "{}/psf_{}.dat".format(sd, sn)

    # --- Rectification ---
    # --------------------
    if config.rectify:
        # add FILTER keyword to headers
        pattern = "vrfnq*{}*sci.fits".format(config.version)
        search = os.path.join(config.frames_directory, pattern)
        raw_frames = glob.glob(search)
        rectify_images(raw_frames, FILTER=config.bandlist[0])
        # make the catalog a little more like what we need
        rectify_catalog(config.raw_catalog, config.initial_catalog, raw_frames[0])
        # Make the PSF store with one gaussian of 3 pixels FWHM
        for band in config.bandlist:
            make_psf_store(config.psfstorefile, fwhm=config.psf_fwhm,
                           amp=config.psf_amp, band=band)


    # --- Make pix and meta stores ---
    # --------------------------------
    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)
    # Make the (empty) metastore
    metastore = MetaStore()

    # --- Find Images ---
    names = find_images(loc=config.frames_directory,
                        pattern="*{}*sci.fits".format(config.version))

    # Fill pixel and metastores
    for n in names:
        pixelstore.add_exposure(n)
        metastore.add_exposure(n)

    bands = list(pixelstore.data.keys())

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)
