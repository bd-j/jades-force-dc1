#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# -----------------------
# --- Filters being run ---
bandlist = ["F090W", "F115W", "F150W", "F200W",
            "F277W", "F335M", "F356W", "F410M", "F444W"]


# -----------------------
# --- Data locations ---
metastorefile = ""
psfstorefile = ""
pixelstorefile = ""
splinedatafile = ""
frames_directory = ""

# ------------------------
# --- Data Types/Sizes ---
pix_dtype = np.int32
meta_dtype = np.float32
super_pixel_size = 8   # number of pixels along one side of a superpixel
nfull = 2048           # number of pixels along one side of a square input frame

# -----------------------
# --- Patch Generation ---
max_active_fraction = 0.1


# -----------------------
# --- HMC parameters ---
nwarm = 250
niter = 100
