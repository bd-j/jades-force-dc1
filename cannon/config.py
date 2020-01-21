#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py - Example configuration script for forcepho runs.
"""

import numpy as np
from argparse import Namespace
config = Namespace()

# -----------
# --- Overall ----
config.log = True

# ---------------
# --- Output -----
config.scene_catalog = "superscene.fits"
config.patchlogfile = "patchlog.dat"

# -----------------------
# --- Filters being run ---
config.bandlist = ["F090W", "F115W", "F150W", "F200W",
                   "F277W", "F335M", "F356W", "F410M", "F444W"]

# -----------------------
# --- Data locations ---
config.storename = "mini-challenge-19-st"
config.pixelstorefile = "stores/pixels_{}.h5".format(config.storename)
config.metastorefile = "stores/meta_{}.dat".format(config.storename)
config.psfstorefile = "stores/psf_jades_ng4.h5"
config.splinedatafile = "stores/sersic_mog_model.smooth=0.0150.h5"
config.frames_directory = "/n/scratchlfs/eisenstein_lab/stacchella/mosaic/st"
config.initial_catalog = "/n/scratchlfs02/eisenstein_lab/bdjohnson/jades_force/data/2019-mini-challenge/source_catalogs/forcepho_table_psf_matched_v5.0.fits"

# ------------------------
# --- Data Types/Sizes ---
config.pix_dtype = np.float32
config.meta_dtype = np.float32
config.super_pixel_size = 8      # number of pixels along one side of a superpixel
config.nside_full = 2048         # number of pixels along one side of a square input frame

# -----------------------
# --- Patch Generation ---
config.max_active_fraction = 0.1
config.maxactive_per_patch = 15

# -----------------------
# --- HMC parameters ---
config.n_warm = 250
config.n_iter = 100
config.n_tune = 100

# ------------------------
# --- PSF information ----
# Used for building PSF store
# generally not necessary
config.mixture_directory = "/Users/bjohnson/Projects/jades_force/data/psf/mixtures"
config.psf_search = "gmpsf*ng4.h5"
