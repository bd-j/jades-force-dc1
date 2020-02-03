#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
import numpy as np
#import matplotlib.pyplot as pl
import logging
import h5py
from scipy.optimize import minimize

# child side
from forcepho.proposal import Proposer
from forcepho.model import GPUPosterior, LogLikeWithGrad
from forcepho.patch import StaticPatch

from utils import Logger, dump_to_h5
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    from patch_conversion import patch_conversion, zerocoords, set_inactive
    maxactive = 15
    patchname = "stores/xpatches/patch_udf_withcat_{}.h5"
    splinedata = "stores/sersic_mog_model.smooth=0.0150.h5"
    psfpath = "stores/xpsf/"
    verbose = False

    patchnum = 159
    patchname = patchname.format(patchnum)

    use_bands = slice(None)
    stamps, scene = patch_conversion(patchname, splinedata, psfpath,
                                     nradii=9, use_bands=use_bands)
    miniscene = set_inactive(scene, [stamps[0], stamps[-1]], nmax=maxactive)
    pra = np.median([s.ra for s in miniscene.sources])
    pdec = np.median([s.dec for s in miniscene.sources])
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))

    patch = StaticPatch(stamps=stamps, miniscene=miniscene, return_residual=True)
    p0 = miniscene.get_all_source_params().copy()
    # last from chain run on ascent for patch 159
    pend = np.array([ 9.75295008e-02,  2.79897763e-01,  1.08403835e-01,  1.52135045e-01,
                      4.21813556e-02,  1.59661042e-01,  1.63261214e-01,  2.08790319e-01,
                      2.03861718e-01, -4.56303975e-04,  7.38754390e-04,  9.58201041e-01,
                     -9.86776402e-01,  4.97859899e+00,  3.02703761e-02,  6.69506029e-02,
                      1.96119388e-01,  1.38156753e-01,  2.16620404e-01,  8.89458965e-02,
                      3.96402219e-01,  2.86238661e-01,  4.47792764e-01,  2.58366487e-01,
                     -7.50394246e-04,  5.41231853e-04,  6.59672153e-01,  1.63510312e-01,
                      1.51872150e+00,  3.80823723e-02,  2.09868476e-01,  4.58055132e-01,
                      2.18298469e-01,  3.11607752e-01,  1.39125697e-01,  7.40847477e-01,
                      7.16032755e-01,  1.07986668e+00,  5.87342472e-01,  9.92837316e-04,
                     -8.91195105e-04,  6.42458102e-01,  5.77076587e-01,  1.08550362e+00,
                      1.13476346e-01,  8.43381427e-02,  3.57579633e-01,  3.43718705e-01,
                      4.76048289e-01,  1.84578080e-01,  7.48954016e-01,  9.12569517e-01,
                      1.18728822e+00,  6.66494872e-01,  1.12853522e-05, -2.89552259e-04,
                      5.31453379e-01,  1.27197483e+00,  1.02464893e+00,  4.68786709e-02,
                      6.64033581e-01,  1.41981116e+00,  6.30341158e-01,  8.24466501e-01,
                      3.07278746e-01,  1.49291870e+00,  1.77601337e+00,  2.70595572e+00,
                      1.58662872e+00,  1.28062919e-03, -1.95026050e-06,  8.55235779e-01,
                      1.88998315e-01,  1.00064189e+00,  1.24386473e-01])


    # --- Copy patch data to device ---
    gpu_patch = patch.send_to_gpu()
    gpu_proposer = Proposer(patch)
    model = GPUPosterior(gpu_proposer, miniscene, name=patchname,
                         verbose=verbose)

    # run the pymc sampling
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)

    model.scene.set_all_source_params(p0)
    pnames = model.scene.parameter_names
    start = dict(zip(pnames, p0))

    z0 = p0.copy()
    dlnp = model.lnprob_grad(z0)
    lnp = model.lnprob(z0)
    ret = model.proposer.evaluate_proposal(z0)

    delta_lnp = 0.01
    delta = delta_lnp / dlnp

    #dlnz = 1e-4
    #delta = z0 * dlnz

    dlnp_num = np.zeros(len(z0), dtype=np.float64)
    for i, dp in enumerate(delta):
        theta = z0.copy()
        theta[i] -= dp
        model.evaluate(theta)
        imlo = model.lnprob(theta)
        theta[i] += 2 * dp
        model.evaluate(theta)
        imhi = model.lnprob(theta)
        dlnp_num[i] = ((imhi - imlo) / (2 * dp))
