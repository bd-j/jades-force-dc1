#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import time

import numpy as np
import argparse
from h5py import File

from storage import ImageNameSet, PixelStore, MetaStore


def find_mosaics(loc="", pattern="*bkgsub.fits"):
    search = os.path.join(os.path.expandvars(loc), pattern)
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f,                        # im
                          f.replace("bkgsub", "err"),  # err
                          "",                       # mask
                          "")                       # bkg
             for f in files]
    return names


def trim_mosaic(filename, outname, super_pixel_size=8,
                **header_kwargs):

    from astropy.nddata import Cutout2D
    from astropy.io import fits
    from astropy.wcs import WCS

    # Load the image and the WCS
    hdul = fits.open(filename)
    hdu = hdul[0]
    wcs = WCS(hdu.header)

    # get the size
    imsize = np.array(hdu.data.shape)
    nsuper = np.floor(imsize / (1.0 * super_pixel_size))
    outsize = tuple((nsuper * super_pixel_size).astype(int))
    # we have to reverse this because ugh
    position = tuple(np.round(imsize / 2.0).astype(int)[::-1])

    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=outsize, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data
    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())
    hdu.header.update(**header_kwargs)

    # Write the cutout to a new FITS file
    hdu.writeto(outname, overwrite=True)
    hdul.close()
    return np.array([outsize[1], outsize[0]])


abmags = {'F115W': 27.5681,
          'F277W': 27.8803,
          'F090W': 27.4525,
          'F356W': 28.0068,
          'F200W': 27.9973,
          'F444W': 28.0647,
          'F150W': 27.814,
          'F335M': 27.0579,
          'F410M': 27.1848}


if __name__ == "__main__":

    from config_mosaic import config
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaics_directory", type=str,
                         default="", help=("location of raw mosaics"))
    parser.add_argument("--store_directory", type=str,
                        default="$SCRATCH/eisenstein_lab/bdjohnson/jades_force/cannon/stores",
                        help="where to put the h5 pixelstore file")
    parser.add_argument("--store_name", type=str,
                        default="mini-challenge-19-mosaic-st")
    parser.add_argument("--frames_directory", type=str,
                        default="$SCRATCH/eisenstein_lab/bdjohnson/jades_force/data/2019-mini-challenge/mosaic/st/trimmed",
                        help="location of trimmed, updated mosaics")
    cpars = vars(parser.parse_args())
    _ = [setattr(config, k, v) for k, v in cpars.items()]

    config.mosaics_directory = os.path.expandvars(config.mosaics_directory)
    config.frames_directory = os.path.expandvars(config.frames_directory)
    config.store_directory = os.path.expandvars(config.store_directory)

    if config.mosaics_directory:
        mfiles = glob.glob(os.path.join(config.mosaics_directory, "*bkgsub.fits"))
        mfiles += glob.glob(os.path.join(config.mosaics_directory, "*err.fits"))
        for mf in mfiles:
            fn = os.path.basename(mf)
            band = fn.split("/")[-1].split("_")[0]
            outfile = os.path.join(config.frames_directory, fn)
            sz = trim_mosaic(mf, outfile, FILTER=band, ABMAG=abmags[band])
            assert np.all(sz == config.nside_full)
        print("trimmed mosaics and copied to {}".format(config.mosaics_directory))


    sd, sn = config.store_directory, config.store_name
    config.pixelstorefile = "{}/pixels_{}.h5".format(sd, sn)
    config.metastorefile = "{}/meta_{}.dat".format(sd, sn)

    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)

    # Make the (empty) metastore
    metastore = MetaStore()
    print("instantiated pixelstore and metastore")

    # --- Find Images ---
    names = find_mosaics(loc=config.frames_directory)
    print("got {} image sets".format(len(names)))
    # Fill pixel and metastores
    for n in names:
        print(n.im)
        pixelstore.add_exposure(n, bitmask=config.bitmask)
        metastore.add_exposure(n)

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)


    # deal with PSFs that now have a different pixel scale for LW
    LW = ["F277W", "F335M", "F356W", "F410M", "F444W"]
    import shutil
    shutil.copy("{}/psf_jades_ng4.h5".format(sd), config.psfstorefile)
    with h5py.File(config.psfstorefile, "a") as mpsf:
        for b in LW:
            pars = mpsf[b]["parameters"]
            pars["xcen"] = pars["xcen"][:] * 2.
            pars["ycen"] = pars["ycen"][:] * 2.
            pars["Cxx"] = pars["Cxx"][:] * 4.
            pars["Cyy"] = pars["Cyy"][:] * 4.
            pars["Cxy"] = pars["Cxy"][:] * 4.

    print("done in {}s".format(time.time() - t))
