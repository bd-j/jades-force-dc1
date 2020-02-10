#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import h5py


__all__ = ["Logger", "dump_to_h5"]


class Logger:

    def __init__(self, name):
        self.name = name
        self.comments = []

    def info(self, message, timetag=None):
        if timetag is None:
            timetag = time.strftime("%y%b%d-%H.%M", time.localtime())

        self.comments.append((message, timetag))

    def serialize(self):
        log = "\n".join([c[0] for c in self.comments])
        return log


def _make_imset(out, paths, name, arrs):
    for i, epath in enumerate(paths):
        try:
            g = out[epath]
        except(KeyError):
            g = out.create_group(epath)

        try:
            g.create_dataset(name, data=np.array(arrs[i]))
        except:
            print("Could not make {}/{} dataset from {}".format(epath, name, arrs[i]))


def dump_to_h5(filename, patch, active=None, fixed=None,
               pixeldatadict={}, otherdatadict={}):
    """Dump patch data and scene data to an HDF5 file
    """
    pix = ["xpix", "ypix", "ierr"]
    meta = ["D", "CW", "crpix", "crval"]
    with h5py.File(filename, "w") as out:

        out.create_dataset("epaths", data=np.array(patch.epaths, dtype="S"))
        out.create_dataset("bandlist", data=np.array(patch.bandlist, dtype="S"))
        out.create_dataset("exposure_start", data=patch.exposure_start)

        for band in patch.bandlist:
            g = out.create_group(band)

        for a in pix:
            arr = getattr(patch, a)
            pdat = np.split(arr, np.cumsum(patch.exposure_N)[:-1])
            _make_imset(out, patch.epaths, a, pdat)

        for a in meta:
            arr = getattr(patch, a)
            _make_imset(out, patch.epaths, a, arr)

        for a, pdat in pixeldatadict.items():
            _make_imset(out, patch.epaths, a, pdat)

        for a, arr in otherdatadict.items():
            out.create_dataset(a, data=arr)

        if active:
            out.create_dataset("active", data=active)
        if fixed:
            out.create_dataset("fixed", data=fixed)
