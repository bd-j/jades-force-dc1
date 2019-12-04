#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from forcepho.patch import Patch

psf_dtype = np.dtype([('gauss_params',dtype,6),('sersic_bin',np.int32)])


JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]


class JadesPatch(Patch):

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

    def process_patch(self, region, sources,
                      bandlist=JWST_BANDS,
                      mass_matrix=None):

        hdrs, superpixels = self.find_pixels(region)

        bands = [hdr["FILTER"] for hdr in hdrs]
        # band_ids must be an int identifier (does not need to be contiguous)
        band_ids = [bandlist.index(b) for b in bands]
        assert (np.diff(band_ids) >= 0).all(), 'Stamps must be sorted by band'
        uniq_bands, n_exp_per_band = np.unique(band_ids, return_counts=True)

        scene = self.set_scene(sources.pars, sources.fluxes[:, uniq_bands],
                               bandlist, splinedata=self.splinedata)

        self.n_bands = len(uniq_bands)       # Number of bands/filters
        self.n_exp = len(hdrs)               # Number of exposures
        self.n_sources = len(scene.sources)  # number of sources

        self.pack_source_metadata(scene)
        self.pack_astrometry(hdrs, scene)
        self.pack_psf(hdrs, scene)

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
            psf_dtype = np.dtype([('gauss_params',dtype,6),('sersic_bin',np.int32)])

        # FIXME: Fill these...
        self.n_psf_per_source = np.empty(self.n_bands, dtype=np.int32)
        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)

        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        s = 0
        for j, hdr in enumerate(hdrs):
            for i, source in enumerate(scene.sources):
                # sources have one set of psf gaussians per exposure
                # length of that set is const in a band, however
                psfparams = get_local_psf(hdr, source)
                N = len(psfparams)
                self.psfgauss[s: (s + N)] = psfparams
                s += N

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
            s = Galaxy(filters=filters.tolist(), splinedata=splinedata, free_sersic=free_sersic)
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


    def find_exposures(self, region):
        """Return a list of headers (dict-like objects of wcs, filter, and
        exposure id) for all exposures that overlap the region.  These should
        be sorted by integer band_id.
        """
        pass

    def find_pixels(self, region):
        pass


class PSFStore:

    def __init__(self, psfdata):
        pass

    def lookup(self, band, x, y, radii=None):
        xp, yp = self.store[band]["pixel_grid"][:]
        dist = np.hypot(x - xp, y - yp)
        choose = dist.argmin()
        pars = self.store[band]["psfs"][choose]
        assert pars.dtype.descr
        return 


def get_local_psf(hdr, source, psfstore):
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
    psf = psfstore.lookup(band, x, y, radii=None)

    return psf
