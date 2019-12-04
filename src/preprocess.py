#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, sys
from collections import namedtuple
import numpy as np
import argparse

parser = argparse.ArgumentParser()

NameSet = namedtuple("Image", ["im", "err", "mask", "bkg"])


def find_images(args):
    pass

def superpixelize(nameset, super_pixel_size=8):

    hdr = fits.getheader(nameset["im"])
    im = fits.getdata(nameset["im"])
    bkg = fits.getdata(nameset["bkg"])
    ierr = 1 / fits.getdata(nameset["err"])
    mask = fits.getdata(nameset["mask"])
    ierr *= (mask == 0)
    im -= bkg

    imsize = np.array(im.shape)
    assert np.all(imsize == 2048)
    nsuper = imsize / super_pixel_size

    flux_images(im, ierr, hdr)

    superpixels = np.array([nsuper[0], nsuper[1], 2 * super_pixel_size**2])
    # slow 
    for i in range(nsuper[0]):
        for j in range(nsuper[1]):
            I = i * super_pixel_size
            J = j * super_pixel_size
            
            superpixels[i, j, 0:super_pixel_size**2] = im[I:I+super_pixel_size, J:J+super_pixel_size].flatten()

    return hdr, superpixels



if __name__ == "__main__":

    args = parser.parse_args()

    names = find_images(args)

    for n in names:
        hdr, superpixels = pack_image(n)
        band = hdr["Filter"]
        h5.create_dataset(pixelpath, data=superpixels)
        h5[pixelpath]


