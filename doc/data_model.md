GPU Data Model
========

Processor Data Model
=======

Input from master: region definition, list of sources (parameters), mass matrix
Output to master:  latest source parameters, latest mass matrix

Input from disk/shared memory: Catalog of exposure metadata, catalog of super-pixel data
Output to disk: initial mass matrix, HMC samples during burn-in, final mass matrix, real HMC samples

Input to GPU:
packed superpixels, packed metadata, packed psfs, packed source sersics.


for send to GPU:

metadata:
-----

 - `D`     [NEXP, NSOURCE, 2, 2]
 - `CW`    [NEXP, NSOURCE, 2, 2]
 - `CRPIX` [NEXP, 2]
 - `CRVAL` [NEXP, 2]
 - `G`     [NEXP]

pixeldata:
--------

 - `xpix`  [total_padded_size]    # total number of pixels
 - `ypix`  [total_padded_size]
 - `data`  [total_padded_size]
 - `ierr`  [total_padded_size]

 - `band_start`     [NBAND]  exposure index that starts a block of bands
 - `band_N`         [NBAND]  number of exposures in a band
 - `exposure_start` [NEXP]   pixel index that starts an exposure
 - `exposure_N`     [NEXP]   number of pixels in an exposure

scenedata:
------

 - `n_sources`    [1]       number of sources in the scene
 - `n_radii`      [1]       number of gaussian radii per source
 - `rad2`         [N_RADII] square of the gaussian radius

 - `n_psf_per_source` [NBAND]  number of PSFGaussians per source in each band
 - `psfgauss`         [NPSFG]  An array of PSFGaussian parameters.  NPSFG=NEXP * N_SOURCES * NPSF_PER_SOURCE
 - `psfgauss_start`   [NEXP]   PSFGaussian index corresponding to the start of each exposure.
