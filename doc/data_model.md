Preprocessed Data
=========

The following files with approximately the following structure are assumed to
exist. Most standard FITS files can be preprocessed to produce valid Meta-Data
and Pixel Data files.  The PSF Data and Sersic splines require more careful
handholding to generate

Meta-Data
-------

A file that can be json loaded into a dictionary of FITS headers. The structure
of the dictionary is:

- `FITSheader = headers[bandname][expID]`

where `bandname` is a string and `expID` is a unique exposure identifier.

The FITS headers must have information that can be converted to a valid WCS by
`astropy.wcs.WCS`. Furthermore, the `"FILTER"` header keyword must be present
and give a string that is the same as `bandname`.

Pixel data
-------

The pixel data is assumed to consist of fluxes and inverse errors
photometrically calibrated (nJy/pixel?) and background subtracted.  Masked
pixels are marked by an inverse-error of 0

This is an HDF file with the structure:

PSF data
------

This is an HDF file with the structure:

For the group corresponding to each band, the attribute `n_psf_per_source` must
be given (as an integer) describing the total number of PSF gaussians (including
for different sersic mixture radii) that is used to represent each source in an exposure.

Sersic Splines
-------

Source Catalog
--------

GPU Data Model
========

A `Patch` object must create the following arrays and then send them to the GPU

metadata
-----

 - `D`     [NEXP, NSOURCE, 2, 2]
 - `CW`    [NEXP, NSOURCE, 2, 2]
 - `CRPIX` [NEXP, 2]
 - `CRVAL` [NEXP, 2]
 - `G`     [NEXP]

pixeldata
--------

 - `xpix`  [total_padded_size]    # total number of pixels
 - `ypix`  [total_padded_size]
 - `data`  [total_padded_size]
 - `ierr`  [total_padded_size]

 - `band_start`     [NBAND]  exposure index that starts a block of bands
 - `band_N`         [NBAND]  number of exposures in a band
 - `exposure_start` [NEXP]   pixel index that starts an exposure
 - `exposure_N`     [NEXP]   number of pixels in an exposure

scenedata
------

 - `n_sources`    [1]       number of sources in the scene
 - `n_radii`      [1]       number of gaussian radii per source
 - `rad2`         [N_RADII] square of the gaussian radius

psfdata:
--------
 - `n_psf_per_source` [NBAND]  number of PSFGaussians per source in each band
 - `psfgauss`         [NPSFG]  An array of PSFGaussian parameters.  NPSFG=NEXP * N_SOURCES * NPSF_PER_SOURCE
 - `psfgauss_start`   [NEXP]   PSFGaussian index corresponding to the start of each exposure.


Processor Data Model
=======

Input from master: region definition, list of sources (parameters), mass matrix
Output to master:  latest source parameters, latest mass matrix

Input from disk/shared memory: Catalog of exposure metadata, catalog of super-pixel data
Output to disk: initial mass matrix, HMC samples during burn-in, final mass matrix, real HMC samples

Input to GPU:
packed superpixels, packed metadata, packed psfs, packed source sersics.


for send to GPU: