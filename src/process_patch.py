#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]


#from forcepho.patch import Patch
#class JadesPatch(Patch):

class JadesPatch:

    def __init__(self,
                 pixelstore,
                 metastore,
                 psfstore,
                 splinedata,
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.int32,
                 super_pixel_size=1):

        self.meta_dtype = meta_dtype
        self.pix_dtype = pix_dtype
        self.splinedata = splinedata

        self.metastore = MetaStore(metastore)
        self.psfstore = PSFStore(psfstore)
        self.pixelstore = PixelStore(pixelstore)

    def process_patch(self, region, sourcecat,
                      bandlist=JWST_BANDS,
                      mass_matrix=None):

        # Find relevant exposures
        hdrs = self.find_exposures(region, bandlist)

        # Get BAND information for the exposures
        bands = [hdr["FILTER"] for hdr in hdrs]
        # band_ids must be an int identifier (does not need to be contiguous)
        band_ids = [bandlist.index(b) for b in bands]
        assert (np.diff(band_ids) >= 0).all(), 'Stamps must be sorted by band'
        uniq_bands, n_exp_per_band = np.unique(band_ids, return_counts=True)

        self.scene = self.set_scene(sourcecat, uniq_bands,
                                    bandlist, splinedata=self.splinedata)

        self.n_bands = len(uniq_bands)       # Number of bands/filters
        self.n_exp = len(hdrs)               # Number of exposures
        self.n_sources = len(self.scene.sources)  # number of sources

        # Pack up all the data for the gpu
        self.pack_source_metadata(self.scene)
        self.pack_astrometry(hdrs, self.scene)
        self.pack_psf(hdrs, self.scene)
        self.pack_pixels(hdrs, region)

    def pack_source_metadata(self, scene, dtype=None):
        """
        We don't actually pack sources in the Patch; that happens
        in a Proposal.  But we do have some global constants related
        to sources, such as the total number of soures and number of
        Sersic radii bins.  So we pack those here.

        Fills in:
        - self.n_sources
        - self.n_radii
        - self.rad2
        """

        if not dtype:
            dtype = self.meta_dtype

        # number of sources in the patch
        self.n_sources = scene.nactive

        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = len(scene.sources[0].radii)

        self.rad2 = np.empty(self.n_radii, dtype=dtype)
        self.rad2[:] = scene.sources[0].radii**2

    def pack_astrometry(self, hdrs, scene, dtype=None):
        """The sources need to know their local astrometric transformation
        matrices (and photometric conversions) in each exposure. We need to
        calculate these from header/meta information and send data to the GPU so
        it can apply the sky-to-pixel transformations to compare with the image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        - self.G
        """
        if not dtype:
            dtype = self.meta_dtype

        self.D = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, 2), dtype=dtype)
        self.G = np.empty((self.n_exp), dtype=dtype)

        for j, hdr in enumerate(hdrs):
            wcs = WCS(hdr)
            self.crval[j] = wcs.crval
            self.crpix[j] = wcs.crpix
            self.G[j] = hdr["data_to_mjy"]
            for i, source in enumerate(scene.sources):
                CW_mat, D_mat = get_transform_mats(source, wcs)
                self.D[j, i] = D_mat
                self.CW[j, i] = CW_mat

    def pack_psf(self, hdrs, scene, dtype=None, psf_dtype=None):
        """Each Sersic radius bin has a number of Gaussians associated with it
        from the PSF. The number of these will be constant in a given band, but
        the Gaussian parameters vary with source and exposure.

        We'll just track the total count across radius bins; the individual
        Gaussians will know which bin they belong to.

        Fills in the following arrays:
        - self.n_psf_per_source   [NBAND]  number of PSFGaussians per source in each band
        - self.psfgauss           [NPSFG]  An array of PSFGaussian parameters
        - self.psfgauss_start     [NEXP]   PSFGaussian index corresponding to the start of each exposure.
        """
        if not dtype:
            dtype = self.meta_dtype
        if not psf_dtype:
            psf_dtype = np.dtype([('gauss_params', dtype, 6), 
                                  ('sersic_bin', np.int32)])

        # FIXME: Fill these...
        self.n_psf_per_source = np.empty(self.n_bands, dtype=np.int32)
        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)

        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        s = 0
        for j, hdr in enumerate(hdrs):
            for i, source in enumerate(scene.sources):
                # sources have one set of psf gaussians per exposure
                # length of that set is const in a band, however
                psfparams = get_local_psf(hdr, source, self.psfstore)
                N = len(psfparams)
                self.psfgauss[s: (s + N)] = psfparams
                s += N

    def pack_pix(self, hdrs, region):
        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

        b, i = 0, 0
        for e, hdr in enumerate(hdrs):
            if e > 0 and hdr["FILTER"] != hdrs[e-1]["FILTER"]:
                b += 1
            self.band_N[b] += 1
            # Get pixel data from this exposure;
            # note these are in super-pixel ordder
            pixdat = self.find_pixels(hdr, region)
            n_pix = len(pixdat[0])
            self.exposure_start[e] = i
            self.exposure_N[e] = n_pix
            data.append(pixdat[0])
            ierr.append(pixdat[1])
            xpix.append(pixdat[2])
            ypix.append(pixdat[3])
            i += n_pix

        self.data = np.concatenate(data)
        assert self.data.shape[0] == i, "pixel data array is not the right shape"
        self.ierr = np.concatenate(ierr)
        self.xpix = np.concatenate(xpix)
        self.ypix = np.concatenate(ypix)
        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]

    def find_exposures(self, region, bandlist):
        """Return a list of headers (dict-like objects of wcs, filter, and
        exposure id) for all exposures that overlap the region.  These should
        be sorted by integer band_id.
        """
        for band in bandlist:
            # TODO: Fill this in
            self.metastore
            

    def find_pixels(self, hdr, region):

        # these are (nsuper, 4) arrays of the full pixel coordinates of the
        # corners of the superpixels
        xc = self.pixelstore.xcorners
        yc = self.pixelstore.ycorners
        # this is just the full coordinates of the lower-left (zeroth) corner of
        # every pixel "contained" within a region
        inx, iny = region.contains(xc, yc, hdr)
        superx = inx / self.pixelstore.super_pixel_size
        supery = iny / self.pixelstore.super_pixel_size

        for i in len(superx):
            pdata = self.pixelstore[expname][superx[i], supery[i], :]
            data = pdata[:nsuper]
            ierr = pdata[nsuper:]
            xpix = superx[i] * self.pixelstore.super_pixel_size + self.pixelstore
            ypix = self.pixelstore


    def set_scene(self, sourcepars, fluxpars, filters,
                  splinedata=None, free_sersic=True):
        """Build a scene from a set of source parameters and fluxes through a
        set of filters.

        Returns
        -------
        scene: Scene object
        """
        sourcepars = sourcepars.astype(np.float)

        # get all sources
        sources = []
        for ii_gal in range(len(sourcepars)):
            gal_id, x, y, q, pa, n, rh = sourcepars[ii_gal]
            #print(x, y, type(pa), pa)
            s = Galaxy(filters=filters.tolist(), splinedata=splinedata,
                       free_sersic=free_sersic)
            s.global_id = gal_id
            s.sersic = n
            s.rh = np.clip(rh, 0.05, 0.10)
            s.flux = fluxpars[ii_gal]
            s.ra = x
            s.dec = y
            s.q = np.clip(q, 0.2, 0.9)
            s.pa = np.deg2rad(pa)
            sources.append(s)

        # generate scene
        scene = Scene(sources)

        return(scene)


class PixelStore:

    def __init__(self, h5file, nfull=2048, super_pixel_size=8):

        self.h5 = h5py.open(h5file)
        if nfull is None:
            self.nfull = np.array(self.h5.attrs["nfull"], dtype=np.int32)
        if super_pixel_size is None:
            self.super_pixel_size = np.array(self.h5.attrs["super_pixel_size"], dtype=np.int32)

        self.nside = self.nfull / self.super_pixel_size
        nsuper = self.nside**2
        ipix = np.arange(len(nsuper))
        # TODO: check this is the proper order
        xp = np.mod(ipix, self.nside) * self.super_pixel_size
        yp = np.floor(ipix / self.nside) * self.super_pixel_size

        # TODO: Check this is the proper order
        self.xcorners = np.array([xp, xp+1, xp, xp+1]).T
        self.ycorners = np.array([yp, yp, yp+1, yp+1]).T

    @property
    def data(self):
        return self.h5


class MetaStore:

    def __init__(self, hdrs=None):
        bands = {}

    def add_hdr(self, hdr):
        band = hdr["FILTER"]
        bands[band][hdr["NAME"]] = hdr



class PSFStore:

    def __init__(self, psfdata):
        self.store = h5py.File(psfdata, "r")

    def lookup(self, band, x, y, radii=None):
        xp, yp = self.store[band]["pixel_grid"][:]
        dist = np.hypot(x - xp, y - yp)
        choose = dist.argmin()
        pars = self.store[band]["psfs"][choose]
        # TODO: assert data dtype is what's required
        #assert pars.dtype.descr
        return pars

    def get_local_psf(self, hdr, source):
        """

        Returns
        --------

        A structured array of psf parameters for a given source in a give band.
        The structure of the array is something like
        amp, xcen, ycen, Cxx, Cyy Cxy, sersic_radius_index
        """
        band = hdr["FILTER"]
        wcs = WCS(hdr)
        x, y = wcs.all_world2pix(source.ra, source.dec)
        psf = self.lookup(band, x, y, radii=None)

        return psf


class CircularRegion:

    def __init__(self, ra, dec, radius):
        self.ra = ra    # degrees
        self.dec = dec  # degrees
        self.radius = radius # degrees of arc

    def contains(self, xcorners, ycorners, hdr):
        """
        xcorners: (nsuper, 4) array 
            the full pixel x coordinates of the corners of superpixels. (x, x+1, x, x+1)
        ycorners
            the full pixel `y` coordinates of the corners of superpixels (y, y, y+1, y+1)
        
        hdr: header of the image including wcs information for the exposure in which to find pixels
        """
        # these oned pixel coordinate arrays should be cached in the
        # pixel store

        # Get the center and radius in pixel coodrinates
        wcs = WCS(hdr)
        xc, yc = wcs.all_world2pix(self.ra, self.dec)
        xr, yr = wcs.all_world2pix(self.ra, self.dec + self.radius)
        r2 = (xc - xr)**2 + (yc - yr)**2

        d2 = (xc - xcorners)**2 + (yc - ycorners)**2
        inreg = np.any(d2 < r2, axis=-1)
        return xcorners[inreg, 0], ycorners[inreg, 0]



def get_transform_mats(source, wcs):
    """Get source specific coordinate transformation matrices CW and D
    """

    # get dsky for step dx, dy = 1, 1
    pos0_sky = np.array([source.ra, source.dec])
    pos0_pix = wcs.wcs_world2pix([pos0_sky], 1)[0]
    pos1_pix = pos0_pix + np.array([1.0, 0.0])
    pos2_pix = pos0_pix + np.array([0.0, 1.0])
    pos1_sky = wcs.wcs_pix2world([pos1_pix], 1)
    pos2_sky = wcs.wcs_pix2world([pos2_pix], 1)

    # compute dpix_dsky matrix
    [[dx_dra, dx_ddec]] = (pos1_pix-pos0_pix) / (pos1_sky-pos0_sky)
    [[dy_dra, dy_ddec]] = (pos2_pix-pos0_pix) / (pos2_sky-pos0_sky)
    CW_mat = np.array([[dx_dra, dx_ddec], [dy_dra, dy_ddec]])

    # compute D matrix
    W = np.eye(2)
    W[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))**-1
    D_mat = 1.0 / 3600.0*np.matmul(W, CW_mat)

    return(CW_mat, D_mat)