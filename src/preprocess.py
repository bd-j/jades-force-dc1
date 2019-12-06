#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, sys

import numpy as np
import argparse

from stores import ImageNameSet, PixelStore, MetaStore

parser = argparse.ArgumentParser()


def find_brants_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/br/"):
    search = loc + "udf_cube_*.slp.flat.fits"
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f, f.replace("slp", "err"), "", f.replace("slp", "bkg")) for f in files]
    return names

def find_sandross_images(loc="/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/mosaics/st/"):
    search = loc + "udf_cube_*.slp.fits"
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(f, f.replace("slp", "err"), "", f.replace("slp", "bkg")) for f in files]
    return names


def flux_calibrate():
    pass

if __name__ == "__main__":

    #args = parser.parse_args()

    names = find_brants_images()

    pixelstorefile = "stores/test.h5"
    nside_full = 2048
    super_pixel_size = 8
    pixelstore = PixelStore(pixelstorefile, nside_full=nside_full,
                            super_pixel_size=super_pixel_size, pix_dtype=np.float)

    #metastore = MetaStore()
    for n in names[:1]:
        pixelstore.add_exposure(n)
        #metastore.add_exposure(n)


        #band = hdr["Filter"]
        #exppath = 
        #pixelstore.create_dataset(expID, data=superpixels)
        #h5[pixelpath]


