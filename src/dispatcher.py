#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from storage import sourcecat_dtype, PAR_COLS
# TODO this should be in this module
from region import CircularRegion


class SuperScene:
    """An object that describes *all* sources in a scene.
    It contains methods for checking out regions, and checking
    them back in while updating their parameters and storing meta-information.

    It generates SuperScene coordinates for all sources (defined as arcseconds
    of latitude and longitude from a the median coordinates of the scene)
    stored in the `scene_coordinates` attribute and builds a KD-Tree based on
    these coordinates for fast lookup.

    A region can be checked out.  The default is to randomly choose a single
    valid source from the catalog and find all sources within some radius of
    that seed source.  The weighting logic for the seeds can be adjusted by
    over-writing the `seed_weight()` method
    """

    def __init__(self, sourcecatfile, statefile="superscene.fits",  # disk locations
                 target_niter=200, maxactive_fraction=0.1,          # stopping criteria
                 maxactive_per_patch=20, nscale=3,                  # patch boundaries
                 boundary_radius=8., maxradius=5., minradius=1):    # patch boundaries

        self.sourcefilename = sourcecatfile
        self.statefilename = statefile
        self.ingest(sourcecatfile)
        self.parameter_columns = PAR_COLS
        #self.inactive_inds = list(range(self.n_sources))
        #self.active_inds = []

        self.n_sources = len(self.sourcecat)
        self.n_active = 0
        self.n_fixed = 0

        self.maxactive_fraction = maxactive_fraction
        self.target_niter = target_niter

        self.maxradius = maxradius
        self.minradius = minradius
        self.maxactive = maxactive_per_patch
        self.boundary_radius = boundary_radius
        self.nscale = 3

        # --- build the KDTree ---
        self.kdt = cKDTree(self.scene_coordinates)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        fits.writeto(self.statefilename, self.sourcecat, overwrite=True)

    @property
    def sparse(self):
        frac = self.n_active * 1.0 / self.n_sources
        return frac < self.maxactive_fraction

    @property
    def undone(self):
        return np.any(self.sourcecat["n_iter"] < self.target_niter)

    def ingest(self, sourcecatfile, bands=None, minrh=0.005):
        cat = fits.getdata(sourcecatfile)
        self.header = fits.getheader(sourcecatfile)
        self.bands = self.header["FILTERS"].split(",")

        self.n_sources = len(cat)
        self.cat_dtype = sourcecat_dtype(bands=self.bands)
        self.sourcecat = np.empty(self.n_sources, dtype=self.cat_dtype)
        for f in cat.dtype.names:
            if f in self.sourcecat.dtype.names:
                self.sourcecat[f][:] = cat[f][:]
        bad = ~np.isfinite(self.sourcecat["rhalf"])
        self.sourcecat["rhalf"][bad] = minrh
        bad = (self.sourcecat["rhalf"] < minrh)
        self.sourcecat["rhalf"][bad] = minrh

        # Store the initial coordinates, which are used to set positional priors
        self.ra0 = cat["ra"][:]
        self.dec0 = cat["dec"][:]
        self.sourcecat["source_index"][:] = np.arange(self.n_sources)

    def sky_to_scene(self, ra, dec):
        """Generate scene coordinates, which are anglular offsets (lat, lon)
        from the median ra, dec in units of arcsec
        """
        c = SkyCoord(ra, dec, unit="deg")
        xy = c.transform_to(self._scene_frame)
        return xy.lon.arcsec, xy.lat.arcsec

    @property
    def scene_coordinates(self):
        """Return cached scene coordinates for all sources, or, if not present,
        build the scene frame and generate and cache the scene coordinates
        before returning them
        """
        try:
            return self._scene_coordinates
        except(AttributeError):
            mra, mdec = np.median(self.sourcecat["ra"]), np.median(self.sourcecat["dec"])
            center = SkyCoord(mra, mdec, unit="deg")
            self._scene_frame = center.skyoffset_frame()
            self._scene_center = (mra, mdec)
            x, y = self.sky_to_scene(self.sourcecat["ra"], self.sourcecat["dec"])
            self.scene_x, self.scene_y = x, y
            self._scene_coordinates = np.array([self.scene_x, self.scene_y]).T
            return self._scene_coordinates

    def checkout_region(self, seed_index=None):

        # Draw a patch center, convert to scene coordinates
        # (arcsec from scene center), and get active and fixed sources
        cra, cdec = self.draw_center(seed_index=seed_index)
        center = self.sky_to_scene(cra, cdec)
        radius, active_inds, fixed_inds = self.get_circular_scene(center)
        # Deal with case that region is invalid
        if radius is None:
            return center, None, None
        region = CircularRegion(cra, cdec, radius / 3600.)
        self.sourcecat["is_active"][active_inds] = True
        self.sourcecat["is_valid"][active_inds] = False
        self.sourcecat["is_valid"][fixed_inds] = False
        self.n_active += len(active_inds)
        self.n_fixed += len(fixed_inds)

        return region, self.sourcecat[active_inds], self.sourcecat[fixed_inds]

    def checkin_region(self, active, fixed, niter, mass_matrix=None):

        # Find where the sources are that are being checked in
        try:
            active_inds = active["source_index"]
            fixed_inds = fixed["source_index"]
        except(KeyError):
            raise

        # replace active source parameters with new parameters
        for f in self.parameter_columns:
            self.sourcecat[f][active_inds] = active[f]

        # update metadata
        self.sourcecat["n_iter"][active_inds] += niter
        self.sourcecat["is_active"][active_inds] = False
        self.sourcecat["n_patch"][active_inds] += 1
        self.sourcecat["is_valid"][active_inds] = True
        self.sourcecat["is_valid"][fixed_inds] = True

        self.n_active -= len(active_inds)
        self.n_fixed -= len(fixed_inds)
        # log which patch and which child ran for each source?

    def get_circular_scene(self, center):
        """
        Parameters
        -------
        center: 2-element array
            Central coordinates in scene units (i.e. arcsec from scene center)

        Returns
        ------
        radius: float
            The radius (in arcsec) from the center that encloses all active sources.

        active_inds: ndarray of ints
            The indices in the supercatalog of all the active sources

        fixed_inds: ndarray of ints
            The indices in the supercatalog of the fixed sources
            (i.e. sources that have some overlap with the radius but are not active)
        """
        # pull all sources within boundary radius
        kinds = self.kdt.query_ball_point(center, self.boundary_radius)
        kinds = np.array(kinds)
        #candidates = self.sourcecat[kinds]

        # check for active sources; if any exist, return None
        # really should do this check after computing a patch radius
        if np.any(self.sourcecat[kinds]["is_active"]):
            return None, None, None

        # sort sources by distance from center in scale-lengths
        rhalf = self.sourcecat[kinds]["rhalf"]
        d = self.scene_coordinates[kinds] - center
        distance = np.hypot(*d.T)
        # This defines a kind of "outer" distnce for each source
        # as the distance plus some number of half-light radii
        # TODO: should use scale radii? or isophotes?
        outer = distance + self.nscale * rhalf
        inner = distance - self.nscale * rhalf

        # Now we sort by outer distance.
        # TODO: *might* want to sort by just distance
        order = np.argsort(outer)

        # How many sources have an outer distance within max patch size
        N_inside = np.argmin(outer[order] < self.maxradius)
        # restrict to <= maxactive.
        N_active = min(self.maxactive, N_inside)

        # set up to maxsources active, add them to active scene
        #active = candidates[order][:N_active]
        active_inds = order[:N_active]
        finds = order[N_active:]
        # define a patch radius:
        # This is the max of active dist + Ns * rhalf, up to maxradius,
        # and at least 1 arcsec
        radius = outer[order][:N_active].max()
        radius = max(self.minradius, radius)

        # find fixed sources, add them to fixed scene:
        #   1) These are up to maxsources sources within Ns scale lengths of the
        #   patch radius defined by the outermost active source
        #   2) Alternatively, all sources within NS scale radii of an active source
        fixed_inds = finds[inner[finds] < radius][:min(self.maxactive, len(finds))]
        # FIXME: make sure at least one source is fixed?
        if len(fixed_inds) == 0:
            fixed_inds = finds[:1]
        return radius, kinds[active_inds], kinds[fixed_inds]

    def draw_center(self, seed_index=None):
        if seed_index:
            k = seed_index
        else:
            k = np.random.choice(self.n_sources, p=self.seed_weight())
        seed = self.sourcecat[k]
        return seed["ra"], seed["dec"]

    def seed_weight(self):

        # just one for inactive, zero if active
        w = (~self.sourcecat["is_active"]).astype(np.float)

        # multiply by something inversely correlated with niter
        # sigmoid ?  This is ~ 0.5 at niter ~ntarget
        # `a` controls how fast it goes to 0 after ntarget
        # `b` shifts the 0.5 weight left (negative) and right of ntarget
        a, b = 20., -1.0
        x = a * (1 - self.sourcecat["n_iter"] / self.target_niter) + b
        w *= 1 / (1 + np.exp(-x))

        return w / w.sum()

    def get_grown_scene(self):
        # option 1
        # grow the tree.
        # stopping criteria:
        #    you hit an active source;
        #    you hit maxactive;
        #    no new sources within tolerance

        # add fixed boundary objects; these will be all objects
        # within some tolerance (d / size) of every active source
        raise NotImplementedError


class MPIQueue:

    def __init__(self, comm, n_children):

        self.comm = comm
        # this is just a list of child numbers
        self.busy = []
        self.idle = list(range(n_children + 1))[1:]
        self.n_children = n_children
        self.parent = 0

    def collect_one(self):
        """Collect from a single child process.  Keeps querying until a child is done
        """
        status = MPI.Status()
        while True:
            # replace explicit loop with source = MPI.ANY_SOURCE?
            for i, (child, req) in enumerate(self.busy):
                stat = self.comm.Iprobe(source=child, tag=MPI.ANY_TAG)
                if stat:
                    # Blocking recieve
                    result = self.comm.recv(source=child, tag=MPI.ANY_TAG,
                                            status=status)
                    ret = self.busy.pop(i)
                    self.idle.append(child)
                    return ret, result

    def submit(self, task, tag=MPI.ANY_TAG):
        child = self.idle.pop(0)
        # Non-blocking send
        req = self.comm.isend(task, dest=child, tag=tag)
        self.busy.append((child, req))
        return child

    def closeout(self):
        while(len(self.idle) > 0):
            c = self.submit(None)