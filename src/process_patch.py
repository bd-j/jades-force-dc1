#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.wcs import WCS

from forcepho.sources import Scene, Galaxy
from stores import MetaStore, PixelStore, PSFStore


JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]


# should match order in patch.cu
PSF_COLS = ["amp", "xcen", "ycen", "Cxx", "Cyy", "Cxy"]

from forcepho.patch import Patch
class JadesPatch(Patch):
#class JadesPatch:

    """This class converts between JADES-like exposure level pixel data,
    meta-data (WCS), and PSF information to the data formats required by the
    GPU-side code.

    Parameters
    ----------

    Important Attributes
    ----------

    bandlist : list of str

    scene : forcepho.Scene()
    """

    def __init__(self,
                 pixelstore="",
                 metastore="",
                 psfstore="",
                 splinedata="",
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 ):

        self.meta_dtype = meta_dtype
        self.pix_dtype = pix_dtype
        self.return_residual = return_residual

        self.splinedata = splinedata
        self.metastore = MetaStore(metastore)
        self.psfstore = PSFStore(psfstore)
        self.pixelstore = PixelStore(pixelstore)

        self.patch_reference_coordinates = np.zeros(2)
        self.wcs_origin = 0

    def build_patch(self, region, sourcecat,
                    allbands=JWST_BANDS):
        """Given a ragion and a source catalog, this method finds and packs up
        all the relevant meta- and pixel-data in a format suitable for transfer
        to the GPU

        Parameters
        ---------
        """
        # --- Find relevant exposures ---
        # The output lists are all of length n_exp and should all be in band
        # order
        meta = self.find_exposures(region, allbands)
        hdrs, wcses, epaths, bands = meta

        # --- Get BAND information for the exposures ---
        # band_ids must be an int identifier (does not need to be contiguous)
        band_ids = [allbands.index(b) for b in bands]
        assert (np.diff(band_ids) >= 0).all(), 'Exposures must be sorted by band'
        uniq_bands, n_exp_per_band = np.unique(band_ids, return_counts=True)
        self.bandlist = [allbands[i] for i in uniq_bands]

        # --- Set the scene ---
        self.scene = self.set_scene(sourcecat, uniq_bands, self.bandlist,
                                    splinedata=self.splinedata)
        # set a reference coordinate near center of scene;
        # subtract this from source coordinates
        self.patch_reference_coordinates = self.zerocoords(self.scene)

        # --- Cache some useful numbers ---
        self.n_bands = len(uniq_bands)       # Number of bands/filters
        self.n_exp = len(hdrs)               # Number of exposures
        self.n_sources = len(self.scene.sources)  # number of sources
        # This gets overwritten by pack_pix
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)
        self.band_N[:] = n_exp_per_band

        # --- Pack up all the data for the gpu ---
        self.pack_source_metadata(self.scene)
        self.pack_astrometry(wcses, self.scene)
        self.pack_fluxcal(hdrs)
        self.pack_psf(bands, wcses, self.scene)

        self.pack_pix(epaths, bands, wcses, region)

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

        Parameters
        ---------
        scene : A forcepho.sources.Scene() instance
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

    def pack_astrometry(self, wcses, scene, dtype=None):
        """The sources need to know their local astrometric transformation
        matrices (and photometric conversions) in each exposure. We need to
        calculate these from header/meta information and send data to the GPU so
        it can apply the sky-to-pixel transformations to compare with the image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        """
        if not dtype:
            dtype = self.meta_dtype

        self.D = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, 2), dtype=dtype)

        for j, wcs in enumerate(wcses):
            self.crval[j] = wcs.wcs.crval - self.patch_reference_coordinates
            self.crpix[j] = wcs.wcs.crpix
            for i, source in enumerate(scene.sources):
                CW_mat, D_mat = get_transform_mats(source, wcs)
                self.D[j, i] = D_mat
                self.CW[j, i] = CW_mat

    def pack_fluxcal(self, hdrs, tweakphot=None, dtype=None):
        """A nominal lux calibrartion has been applied to all images,
        but here we allow for tweaks to the flux calibration.

        Fills in the following array:
        - self.G
        """
        if not dtype:
            dtype = self.meta_dtype

        self.G = np.ones((self.n_exp), dtype=dtype)
        if not tweakphot:
            return
        else:
            for j, hdr in enumerate(hdrs):
                self.G[j] = hdr[tweakphot]

    def pack_psf(self, bands, wcses, scene, dtype=None, psf_dtype=None):
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
            psf_dtype = np.dtype([(c, dtype) for c in PSF_COLS] +
                                 [("sersic_bin", np.int32)])

        # Get number of gaussians per source for each band
        npsf_per_source = [self.psfstore.data[b].attrs["n_psf_per_source"]
                           for b in self.bandlist]
        self.n_psf_per_source = np.empty(self.n_bands, dtype=np.int32)
        self.n_psf_per_source[:] = np.array(npsf_per_source)

        # Make array for PSF parameters and index into that array
        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)
        n_psfgauss = (self.n_psf_per_source * self.band_N * self.n_sources).sum()
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        s = 0
        for e, wcs in enumerate(wcses):
            self.psfgauss_start[e] = s
            for i, source in enumerate(scene.sources):
                # sources have one set of psf gaussians per exposure
                # length of that set is const in a band, however
                psfparams = self.psfstore.get_local_psf(band=bands[e], source=source)
                N = len(psfparams)
                self.psfgauss[s: (s + N)] = psfparams
                s += N
        assert s == n_psfgauss

    def pack_pix(self, epaths, bands, wcses, region, dtype=None):
        """We have super-pixel data in individual exposures that we want to
        pack into concatenated 1D pixel arrays.

        As we go, we want to build up the index arrays that allow us to find
        an exposure in the 1D arrays.

        Fills the following arrays:
        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start     [NBAND] exposure index corresponding to the start of each band
        - self.band_N         [NBAND] number of exposures in each band
        - self.exposure_start [NEXP]  pixel index corresponding to the start of each exposure
        - self.exposure_N     [NEXP]  number of pixels (including warp padding) in each exposure
        """
        if not dtype:
            dtype = self.pix_dtype

        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

        # wish I knew how many superpixels there would be;
        # could set an upper bound based on ragion area * nexp
        #total_padded_size = region.area * n_exp
        #self.xpix = np.zeros(total_padded_size, dtype=dtype)
        #self.ypix = np.zeros(total_padded_size, dtype=dtype)
        #self.data = np.zeros(total_padded_size, dtype=dtype)  # value (i.e. flux) in pixel.
        #self.ierr = np.zeros(total_padded_size, dtype=dtype)  # 1/sigma
        data, ierr, xpix, ypix = [], [], [], []

        b, i = 0, 0
        for e, wcs in enumerate(wcses):
            # Get pixel data from this exposure;
            # NOTE: these are in super-pixel order
            # TODO: Handle exposures with no valid pixels gracefully,
            # or make sure they are not in the original list of headers
            pixdat = self.find_pixels(epaths[e], wcs, region)
            # use size instead of len here because we are going to flatten everything
            n_pix = pixdat[0].size
            if e > 0 and bands[e] != bands[e - 1]:
                b += 1
            self.band_N[b] += 1
            self.exposure_start[e] = i
            self.exposure_N[e] = n_pix
            data.append(pixdat[0])
            ierr.append(pixdat[1])
            xpix.append(pixdat[2])
            ypix.append(pixdat[3])
            i += n_pix

        #print(i)
        # FIXME set the dtype explicitly here
        #total_padded_size = i
        #self.xpix = np.zeros(total_padded_size, dtype=dtype)
        #self.ypix = np.zeros(total_padded_size, dtype=dtype)
        #self.data = np.zeros(total_padded_size, dtype=dtype)  # value (i.e. flux) in pixel.
        #self.ierr = np.zeros(total_padded_size, dtype=dtype)  # 1/sigma
        #np.concatenate(data, out=self.data)
        self.data = np.concatenate(data)#.reshape(-1).astype(dtype)
        #assert self.data.shape[0] == i, "pixel data array is not the right shape"
        self.ierr = np.concatenate(ierr).reshape(-1).astype(dtype)
        self.xpix = np.concatenate(xpix).reshape(-1).astype(dtype)
        self.ypix = np.concatenate(ypix).reshape(-1).astype(dtype)
        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]

    def find_exposures(self, region, bandlist, imsize=(2048, 2048)):
        """Return a list of headers (dict-like objects of wcs, filter, and
        exposure id) and exposureIDs for all exposures that overlap the region.
        These should be sorted by integer band_id.
        """
        super_corners = self.pixelstore.superpixel_corners()
        bra, bdec = region.bounding_box

        epaths, bands, hdrs, wcses = [], [], [], []
        for band in bandlist:
            # TODO: Fill this in
            for expID in self.metastore.wcs[band].keys():
                epath = "{}/{}".format(band, expID)
                wcs = self.metastore.wcs[band][expID]
                # Check region bounding box has a corner in the exposure.
                # NOTE: If bounding box entirely contains image this might fail
                bx, by = wcs.all_world2pix(bra, bdec, self.wcs_origin)
                inim = np.any((bx > 0) & (bx < imsize[0]) &
                              (by > 0) & (by < imsize[1]))
                if inim:
                    # check in more detail
                    sx, sy = region.contains(super_corners[..., 0],
                                             super_corners[..., 1], wcs,
                                             origin=self.wcs_origin)
                    if len(sx) == 0:
                        # skip this exposure
                        continue
                    wcses.append(wcs)
                    epaths.append(epath)
                    bands.append(band)
                    hdrs.append(self.metastore.headers[band][expID])
        return hdrs, wcses, epaths, bands


    def find_pixels(self, epath, wcs, region):
        """Find all super-pixels in an image described by `hdr` that are within
        a given region, and return lists of the super-pixel data

        Parameters
        -----------

        Returns
        ------------

        data, ierr, x, y

        """
        s2 = self.pixelstore.super_pixel_size**2
        # this is a (nside, nside, 4, 2) array of the full pixel coordinates of
        # the corners of the superpixels:
        corners = self.pixelstore.superpixel_corners()
        # this returns the superpixel coordinates of every pixel "contained"
        # within a region:
        sx, sy = region.contains(corners[..., 0], corners[..., 1], wcs, origin=self.wcs_origin)
        data = self.pixelstore.data[epath+"/data"][:]
        xpix = self.pixelstore.xpix[sx, sy, :]
        ypix = self.pixelstore.ypix[sx, sy, :]

        return data[sx, sy, :s2], data[sx, sy, s2:], xpix, ypix

    def set_scene(self, sourcepars, band_ids, filters,
                  splinedata=None, free_sersic=True):
        """Build a scene from a set of source parameters and fluxes through a
        set of filters.

        Parameters
        ---------
        sourcepars : structured ndarray
            each row is a source.  It should unpack as:
            id, ra, dec, q, pa, n, rh, flux, flux_unc

        band_ids : list of ints or slice
            The elements of the flux array in `sourcepars` corresponding to
            the given filters.

        filters : list of strings
            The list of the band names that are being used for this patch

        splinedata : string
            Path to the HDF5 file containing spline information.
            This should be the actual spline information...

        Returns
        -------
        scene: Scene object
        """
        #sourcepars = sourcepars.astype(np.float)

        # get all sources
        sources = []
        for ii, pars in enumerate(sourcepars):
            gid, x, y, q, pa, n, rh, flux, unc = pars
            s = Galaxy(filters=filters, splinedata=splinedata,
                       free_sersic=free_sersic)
            s.global_id = gid
            s.sersic = n
            s.rh = np.clip(rh, 0.05, 0.10)
            s.flux = flux[band_ids]
            s.ra = x
            s.dec = y
            s.q = np.clip(q, 0.2, 0.9)
            s.pa = np.deg2rad(pa)
            sources.append(s)

        # generate scene
        scene = Scene(sources)

        return(scene)

    def zerocoords(self, scene, sky_zero=None):
        """Reset (in-place) the celestial zero point of the image metadata and
        the source coordinates to avoid catastrophic cancellation errors in
        coordinates when using single precision.

        Parameters
        ----------
        scene:
            A Scene object, where each source has the attributes `ra`, `dec`,

        sky_zero: optional, 2-tuple of float64
            The (ra, dec) values defining the new central coordinates.  These
            will be subtracted from the relevant source and stamp coordinates.
            If not given, the median coordinates of the scene will be used.
        """
        if not sky_zero:
            zra = np.median([s.ra for s in scene.sources])
            zdec = np.median([s.dec for s in scene.sources])
            sky_zero = np.array([zra, zdec])

        zero = np.array(sky_zero)
        self.patch_reference_coordinates = zero
        for source in scene.sources:
            source.ra -= zero[0]
            source.dec -= zero[1]

        return zero
        # now subtract from all pixel metadata
        #self.crval -= zero[None, :]


class CircularRegion:

    """An object that defines a circular region in celestial coordinates.  It
    contains methods that give a simple bounding box in celestial coordinates,
    and that can determine, given a wcs,  whether a set of pixel corners
    (in x, y) are contained within a region

    Parameters
    ----------

    ra : float
        Right Ascension of the center of the circle.  Degrees

    dec : float
        The Declination of the center of the circle.  Degrees

    radius : float
        The radius of the region, in degrees of arc.
    """

    def __init__(self, ra, dec, radius):
        self.ra = ra          # degrees
        self.dec = dec        # degrees
        self.radius = radius  # degrees of arc

    def contains(self, xcorners, ycorners, wcs, origin=0):
        """
        xcorners: (nsuper, nsuper, 4) ndarray
            the full pixel x coordinates of the corners of superpixels. (x, x+1, x, x+1)
        ycorners : (nsuper, nsuper, 4) ndarray
            the full pixel `y` coordinates of the corners of superpixels (y, y, y+1, y+1)

        hdr: header of the image including wcs information for the exposure in which to find pixels
        """
        # Get the center and radius in pixel coodrinates
        xc, yc = wcs.all_world2pix(self.ra, self.dec, origin)
        xr, yr = wcs.all_world2pix(self.ra, self.dec + self.radius, origin)
        r2 = (xc - xr)**2 + (yc - yr)**2
        d2 = (xc - xcorners)**2 + (yc - ycorners)**2
        inreg = np.any(d2 < r2, axis=-1)
        return np.where(inreg)

    @property
    def bounding_box(self):
        dra = self.radius / np.cos(np.deg2rad(self.dec))
        ddec = self.radius
        corners = [(self.ra - dra, self.dec - ddec),
                   (self.ra + dra, self.dec - ddec),
                   (self.ra + dra, self.dec + ddec),
                   (self.ra - dra, self.dec + ddec)]
        return np.array(corners).T


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
