#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mc.py - mthods for hmc with pymc3.  This should (almost) all go in the forcepho.fitting module
"""

import numpy as np
from forcepho.model import LogLikeWithGrad

import theano
import pymc3 as pm
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
import theano.tensor as tt
theano.gof.compilelock.set_lock_status(False)


def prior_bounds(scene, pos_prior=0.1/3600., flux_factor=5, parname="proposal"):
    """Generate a pymc3 prior distribution for the scene parameter proposal

    Parameters
    ----------
    scene : forcepho.sources.Scene() instance
        The scene.

    pos_prior : float, optional (default: 0.1/3600)
        The positional prior half-width, in degrees, as a coordinate distance
        from the input scene positions.

    flux_factor : float, optional (default: 5)
        The upper limit for the flux will be this factor times the input scene
        fluxes.

    parname : string, optional (default: "proposal")
        The name of the output distribution

    Returns
    -------
    z0 : pymc3.Distribution
        A (vector) distribution describing the prior on the parameters.  This
        will generally be a multivariate uniform with different upper and
        lower bounds in each dimension.

    start : dict
        A dictionary keyed by `parname` that gives the starting value for the
        parameter.
    """
    pnames = scene.parameter_names

    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)
    lower = [s.nband * [0.] +
             [s.ra - pos_prior, s.dec - pos_prior,
              0.3, -np.pi/1.5, sersic_range[0], rh_range[0]]
             for s in scene.sources]
    upper = [(np.array(s.flux) * flux_factor).tolist() +
             [s.ra + pos_prior, s.dec + pos_prior,
              1.0, np.pi/1.5, sersic_range[-1], rh_range[-1]]
             for s in scene.sources]
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)
    #z0 = [pm.Uniform(p, lower=l, upper=u) 
    #      for p, l, u in zip(pnames, lower, upper)]
    z0 = [pm.Uniform(parname, lower=lower, upper=upper, shape=lower.shape)]
    s0 = scene.get_all_source_params()
    start = {parname: s0}

    return z0


def simple_run(model, p0, n_iter=50, n_warm=100, prior_bounds=None):

    # -- Launch HMC ---
    # wrap the loglike and grad in theano tensor ops
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)

    # Get upper and lower bounds for variables
    #lower, upper = prior_bounds(model.proposer.patch.scene)
    #print(lower.dtype, upper.dtype)
    model.scene.set_all_parameters(p0)
    pnames = model.scene.parameter_names
    start = dict(zip(pnames, p0))

    # The pm.sample() method below will draw an initial theta, 
    # then call logl.perform and logl.grad multiple times
    # in a loop with different theta values.
    with pm.Model() as opmodel:
        # set priors for each element of theta
        z0 = prior_bounds(model.scene)
        theta = tt.as_tensor_variable(z0)
        # instantiate target density and start sampling.
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=n_iter, tune=n_warm, start=start,
                          cores=1, progressbar=False,
                          discard_tuned_samples=True)
    return trace, None


def run_pymc3(model, p0, iterations, init_cov=None, prior_bounds=None):

    # -- Launch HMC ---
    # wrap the loglike and grad in theano tensor ops
    model.proposer.patch.return_residuals = False
    logl = LogLikeWithGrad(model)

    # Get upper and lower bounds for variables
    #lower, upper = prior_bounds(model.proposer.patch.scene)
    #print(lower.dtype, upper.dtype)
    pnames = model.scene.parameter_names
    start = dict(zip(pnames, p0))

    # Define the windows used to tune the mass matrix.
    k = np.floor(np.log2((config.n_tune - config.n_warm) / config.n_start))
    nwindow = config.n_start * 2 ** np.arange(k)
    nwindow = np.append(nwindow, config.n_tune - config.n_warm - np.sum(nwindow))
    nwindow = nwindow.astype(int)

    # The pm.sample() method below will draw an initial theta,
    # then call logl.perform and logl.grad multiple times
    # in a loop with different theta values.
    with pm.Model() as opmodel:

        # Set priors for each element of theta.
        z0 = prior_bounds(model.scene)
        theta = tt.as_tensor_variable(z0)

        # Instantiate target density.
        pm.DensityDist('likelihood', lambda v: logl(v),
                        observed={'v': theta})

        # Tune mass matrix.
        start, burnin = None, None
        for steps in iterations.nwindow:
            step = get_step_for_trace(init_cov=init_cov, trace=burnin)
            burnin = pm.sample(start=start, tune=steps, draws=2, step=step,
                               compute_convergence_checks=False,
                               progressbar=False, cores=1,
                               discard_tuned_samples=False)
            start = [t[-1] for t in burnin._straces.values()]
        step = get_step_for_trace(init_cov=init_cov, trace=burnin)

        # Sample with tuned mass matrix.
        trace = pm.sample(draws=iterations.n_iter, tune=iterations.n_warm,
                          step=step, start=start,
                          progressbar=False, cores=1,
                          discard_tuned_samples=True)

        return trace, step


def get_step_for_trace(init_cov=None, trace=None, model=None,
                       regularize_cov=False,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    """
    Construct an estimate of the mass matrix based on the sample covariance,
    which is either provided directly via `init_cov` or generated from a
    `MultiTrace` object from PyMC3. This is then used to initialize a `NUTS`
    object to use in `sample`.
    """

    # ???
    model = pm.modelcontext(model)

    # If no trace or covariance is provided, just use the identity.
    if trace is None and init_cov is None:
        potential = QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **nuts_kwargs)

    # If the trace is provided, loop over samples
    # and convert to the relevant parameter space.
    if trace is not None:
        samples = np.empty((len(trace) * trace.nchains, model.ndim))
        i = 0
        for chain in trace._straces.values():
            for p in chain:
                samples[i] = model.bijection.map(p)
                i += 1

        # Compute the sample covariance.
        cov = np.cov(samples, rowvar=False)

        # Stan uses a regularized estimator for the covariance matrix to
        # be less sensitive to numerical issues for large parameter spaces.
        if regularize_cov:
            N = len(samples)
            cov = cov * N / (N + regular_window)
            diags = np.diag_indices_from(cov)
            cov[diags] += ((regular_variance * regular_window)
                           / (N + regular_window))
    else:
        # Otherwise, just copy `init_cov`.
        cov = np.array(init_cov)

    # Use the sample covariance as the inverse metric.
    potential = QuadPotentialFull(cov)

    return pm.NUTS(potential=potential, **nuts_kwargs)
