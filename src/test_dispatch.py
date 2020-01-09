#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time
import numpy as np
#import matplotlib.pyplot as pl
from mpi4py import MPI

import logging

from argparse import Namespace
from default_config import config

# parent side
from dispatcher import SuperScene, MPIQueue


# child side
def do_work(region, active, fixed, mm):
    # pretend to do work
    time.sleep(1)
    result = Namespace()
    result.niter = 75
    result.active = active
    result.fixed = fixed
    return result


#log = logging.getLogger(__name__)
#_VERBOSE = 10


if __name__ == "__main__":

    # load parameters
    config.initial_catalog = "/Users/bjohnson/Projects/jades_force/data/2019-mini-challenge/source_catalogs/photometry_table_psf_matched_v1.0.fits"
    config.patchlogfile = "patchlog.dat"

    # MPI communicator
    comm = MPI.COMM_WORLD
    child = comm.Get_rank()
    parent = 0
    status = MPI.Status()

    config.nchildren = comm.Get_size() - 1

    if (not child):
        tstart = time.time()
        patchcat = {}
        # Make Queue
        queue = MPIQueue(comm, config.nchildren)
        with SuperScene(config.initial_catalog) as sceneDB:
            # LOOP
            patchid = 0
            while True:
                # Generate patch proposals and send to idle children
                work_to_do = ((len(queue.idle) > 0)
                               & sceneDB.sparse
                               & sceneDB.undone
                              )
                print(work_to_do)
                #work_to_do = True
                while work_to_do:
                    n, active = 0, None
                    # keep asking for patches until a valid one is found
                    while active is None:
                        region, active, fixed = sceneDB.checkout_region()
                        mass = None  # TODO: this should be returned by the superscene
                        n += 1
                        patchid += 1
                    # construct the task
                    chore = (region, (active, fixed, mass))
                    patchcat[patchid] = {"ra": region.ra,
                                         "dec": region.dec,
                                         "radius": region.radius,
                                         "sources": active["source_index"].tolist()}
                    # submit the task
                    assigned_to = queue.submit(chore, tag=patchid)
                    # TODO: Log the submission
                    #log.log(_VERBOSE, "Sent {} with {} to child {}".format(patchid, region.ra, assigned_to))
                    print("Sent patch {} with {} active sources and ra {} to child {}".format(patchid, len(active), region.ra, assigned_to))
                    # Check if we can submit to more children
                    work_to_do = ((len(queue.idle) > 0)
                                  & sceneDB.sparse
                                  & sceneDB.undone
                                  )

                # collect from a single child and set it idle
                c, result = queue.collect_one()
                #print(result)
                # TODO: Log the collection
                sceneDB.checkin_region(result.active, result.fixed,
                                       result.niter, mass_matrix=None)

                # End criterion
                end = len(queue.idle) == queue.n_children
                if end:
                    ttotal = time.time() - tstart
                    print("finished in {}s".format(ttotal))
                    break

        import json
        with open(config.patchlogfile, "w") as f:
            json.dump(patchcat, f)
        queue.closeout()

    elif child:
        # Event Loop
        status = MPI.Status()
        while True:
            # probe: do we need to do this?

            # wait or receive
            # TODO: irecv ?
            task = comm.recv(source=parent, tag=MPI.ANY_TAG,
                             status=status)
            # if shutdown break and quit
            if task is None:
                break

            region, cats = task
            active, fixed, mm = cats
            patchid = status.tag
            #log.log(_VERBOSE, "Child {} received {} with tag {}".format(child, region.ra, status.tag))
            print("Child {} received {} with tag {}".format(child, region.ra, patchid))
            # pretend we did something
            result = do_work(region, active, fixed, mm)
            print(result.active["n_iter"].min(), result.active["n_iter"].max())
            # develop the payload
            payload = result

            # send to parent, free GPU memory
            # TODO: isend?
            comm.ssend(payload, parent, status.tag)
            #patcher.free()



def simple_test(config):

    sceneDB = SuperScene(config.initial_catalog)

    ntest = 100
    regions, active, fixed, count = [], [], [], []

    for i in range(ntest):
        n, a = 0, None
        while a is None:
            r, a, f = sceneDB.checkout_region()
            n += 1
        regions.append(r)
        active.append(a)
        fixed.append(f)
        count.append(n)