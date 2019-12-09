#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import namedtuple
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS

ImageNameSet = namedtuple("Image", ["im", "err", "mask", "bkg"])
SOURCECAT_DTYPE = None
EXP_FMT = "{}/{}"


def header_to_id(hdr, nameset):
    band = hdr["FILTER"]
    expID = os.path.basename(nameset.im).replace(".fits", "")
    return band, expID


class PixelStore:
    """Organization of the pixel data store is

    `bandID/expID/data`

    where `data` is an array of shape (nsuper, nsuper, 2*super_pixel_size**2) 
    The first half of the trailing dimension is the pixel flux information,
    while the second half is the ierr information. Each dataset has attributes
    that describe the nominal flux calibration that was applied and some
    information about the subtracted background, mask, etc.
    """

    def __init__(self, h5file, nside_full=2048, super_pixel_size=8,
                 pix_dtype=np.float32):

        self.h5file = h5file
        self.nside_full = nside_full
        self.super_pixel_size = super_pixel_size
        self.pix_dtype = pix_dtype

        self.nside_super = self.nside_full / self.super_pixel_size
        self.xpix, self.ypix = self.pixel_coordinates()

    def superpixel_corners(self, imsize=None):
        if not imsize:
            xpix, ypix = self.xpix, self.ypix
        else:
            # This is inefficient
            xpix, ypix = self.pixel_coordinates(imsize=imsize)
        # Full image coordinates of the super pixel corners
        xx = xpix[:, :, 0], xpix[:, :, -1]
        yy = ypix[:, :, 0], ypix[:, :, -1]
        # FIXME: get all 4 corners

    def pixel_coordinates(self, imsize=None):
        if not imsize:
            imsize = [self.nside_full, self.nside_full]
        # NOTE: the order swap here for x, y
        yy, xx = np.meshgrid(np.arange(imsize[1]), np.arange(imsize[0]))
        packed = self.superpixelize(xx, yy)
        xpix = packed[:, :, :self.super_pixel_size**2]
        ypix = packed[:, :, self.super_pixel_size**2:]
        return xpix, ypix

    def add_exposure(self, nameset):
        """Add an exposure to the pixel data store, including super-pixel
        ordering.

        Parameters
        -------------

        nameset : NamedTuple with attributes `im`, `err`, `bkg`, `mask`
            A set of names (including path) for a given exposure.
        """
        # Read the header and set identifiers
        hdr = fits.getheader(nameset.im)
        band, expID = header_to_id(hdr, nameset)

        # Read data and perform basic operations
        # NOTE: we transpose to get a more familiar order where the x-axis
        # (NAXIS1) is the first dimension and y is the second dimension.
        im = np.array(fits.getdata(nameset.im)).T
        bkg = np.array(fits.getdata(nameset.bkg)).T
        ierr = 1 / np.array(fits.getdata(nameset.err)).T
        mask = ~(np.isfinite(ierr) & np.isfinite(im))
        if nameset.mask:
            mask *= np.array(fits.getdata(nameset.mask)).T
        ierr *= (mask == 0)
        im -= bkg
        # this does nominal flux calibration of the image.
        # Returns the calibration factor applied
        fluxconv = self.flux_calibration(hdr)

        # Superpixelize
        imsize = np.array(im.shape)
        assert np.all(np.mod(imsize, self.super_pixel_size) == 0)
        if np.any(imsize != self.nside_full):
            # In principle this can be handled, but for now we assume all
            # images are the same size
            raise ValueError("Image is not the expected size")
        nsuper = imsize / self.super_pixel_size
        superpixels = self.superpixelize(im, ierr)

        # Put into the HDF5 file; note this opens and closes the file
        with h5py.File(self.h5file, "a") as h5:
            path = "{}/{}".format(band, expID)
            try:
                exp = h5.create_group(path)
            except(ValueError):
                del h5[path]
                print("deleted existing data for {}".format(path))
                exp = h5.create_group(path)
            pdat = exp.create_dataset("data", data=superpixels)
            pdat.attrs["counts_to_flux"] = fluxconv
            for i, f in enumerate(nameset._fields):
                pdat.attrs[f] = nameset[i]

    def superpixelize(self, im, ierr, pix_dtype=None):

        super_pixel_size = self.super_pixel_size
        s2 = super_pixel_size**2
        nsuper = (np.array(im.shape) / super_pixel_size).astype(int)
        if not pix_dtype:
            pix_dtype = self.pix_dtype
        superpixels = np.empty([nsuper[0], nsuper[1], 2 * super_pixel_size**2],
                               dtype=pix_dtype)
        # slow
        for ii in range(nsuper[0]):
            for jj in range(nsuper[1]):
                I = ii * super_pixel_size
                J = jj * super_pixel_size
                # TODO: check x,y ordering
                superpixels[ii, jj, :s2] = im[I:(I + super_pixel_size),
                                              J:(J + super_pixel_size)].flatten()
                superpixels[ii, jj, s2:] = ierr[I:(I + super_pixel_size),
                                                J:(J + super_pixel_size)].flatten()
        return superpixels

    def flux_calibration(self, hdr):
        return 1.0

    # Need better file handle treatment here.
    # should test for open file handle and return it, otherwise create and cache it
    @property
    def data(self):
        return h5py.File(self.h5file, "r", swmr=True)


class MetaStore:

    def __init__(self, metastorefile=None):
        if not metastorefile:
            self.headers = {}
        else:
            self.headers = self.read_from_file(metastorefile)
            self.populate_wcs()

    def populate_wcs(self):
        self.wcs = {}
        for band in self.headers.keys():
            self.wcs[band] = {}
            for expID, hdr in self.headers[band].items():
                self.wcs[band][expID] = WCS(hdr)

    def add_exposure(self, nameset):
        hdr = fits.getheader(nameset.im)
        band, expID = header_to_id(hdr, nameset)
        if band not in self.headers:
            self.headers[band] = {}
        self.headers[band][expID] = hdr

    def write_to_file(self, filename):
        import json
        hstrings = {}
        for band in self.headers.keys():
            hstrings[band] = {}
            for expID, hdr in list(self.headers[band].items()):
                hstrings[band][expID] = hdr.tostring()
        with open(filename, "w") as f:
            json.dump(hstrings, f)

    def read_from_file(self, filename):
        H = fits.Header()
        import json
        headers = {}
        with open(filename, "r") as f:
            sheaders = json.load(f)
        for band in sheaders.keys():
            headers[band] = {}
            for expID, h in sheaders[band].items():
                headers[band][expID] = H.fromstring(h)

        return headers


class PSFStore:
    """Assumes existence of a file with the following structure

    band/pixel_grid
    band/psfs

    where psfs is a dataset like
      psfs = np.zeros(nloc, nradii, ngauss, dtype=pdt)
      pdt = np.dtype([('gauss_params', np.float, 6),
                      ('sersic_bin', np.int32)])
    and the order of gauss_params is given in patch.cu; amp, x, y, Cxx, Cyy, Cxy

    In principle ngauss can depend on i_radius
    """

    def __init__(self, h5file):
        self.h5file = h5file
        self.filehandle = h5py.File(h5file, "r")

    def lookup(self, band, xy=None):
        """Returns a array of shape (nradii x ngauss,) with dtype
        """
        try:
            x, y = xy
            xp, yp = self.data[band]["detector_locations"][:]
            dist = np.hypot(x - xp, y - yp)
            choose = dist.argmin()
        except:
            choose = 0
        pars = self.data[band]["parameters"][choose]
        # TODO: assert data dtype is what's required for JadesPatch
        #assert pars.dtype.descr
        return pars

    def get_local_psf(self, band="F090W", source=None, wcs=None):
        """
        Returns
        --------
        A structured array of psf parameters for a given source in a give band.
        The structure of the array is something like
        (amp, xcen, ycen, Cxx, Cyy Cxy, sersic_radius_index)
        There are npsf_per_source rows in this array.
        """
        if wcs:
            xy = wcs.all_world2pix(source.ra, source.dec)
        else:
            xy = None
        psf = self.lookup(band, xy=xy)

        return psf

    @property
    def data(self):
        return self.filehandle