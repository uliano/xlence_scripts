#!/usr/bin/env python

import mdtraj as md
import numpy as np
import os
import pickle
import argparse


def do_hbonds(trajectory, topology=None, verbose=True, distance_cutoff=0.35, angle_cutoff=120,
              start_time=-np.inf, end_time=np.inf, skip=1, chunk_size=100, periodic=True,
              exclude_waters=False, sidechain_only=False):
    if chunk_size % skip:
        raise ValueError('do_hbonds: chunk_size must be an integer multiple of step')
    if topology is None:
        base, _ = os.path.splitext(trajectory)
        topology = base + '.pdb'
    hbonds = []
    for trj in md.iterload(trajectory, top=topology, chunk=chunk_size):
        if trj.time[0] >= end_time or trj.time[-1] <= start_time:
            continue

        indices = np.argwhere((trj.time >= start_time) & (trj.time <= end_time))
        keep = list(range(0, chunk_size, skip))
        ok_indices = [index[0] for index in indices if index in keep]
        skip_range = range(min(ok_indices), max(ok_indices) + skip, skip)
        trj_ok = trj[skip_range]

        for i in range(len(trj_ok)):

            frame = trj_ok[i]
            time = frame.time[0]
            if verbose:
                print(f'HBONDS - Processing time: {time}', end='\r')
            the_hbonds = md.baker_hubbard(frame, periodic=periodic, distance_cutoff=distance_cutoff,
                                          angle_cutoff=angle_cutoff, exclude_water=exclude_waters,
                                          sidechain_only=sidechain_only, freq=0)
            hbonds.append((time, the_hbonds))
    if verbose:
        print()
        print('Done.')
    return hbonds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", help='The trajectory')
    parser.add_argument("-top", help='The topology. Defaults to "traj_name.pdb"')
    parser.add_argument("-cutoff", type=float, default=0.35, help='cutoff distance')
    parser.add_argument("-cutoff_angle", type=float, default=120.0)
    parser.add_argument("-periodic", default=True)
    parser.add_argument("-exclude_waters", default=True)
    parser.add_argument("-sidechain_only", default=False)
    parser.add_argument("-start_time", type=float, help="(ps) process only frames after this time")
    parser.add_argument("-end_time", type=float, help="(ps) process only frames before this time")
    parser.add_argument("-skip", type=int, default=1, help="process only every skip frame")
    parser.add_argument("-chunk_size", type=int, default=100, help='load in memory only "chunk_size" frames at a time')
    parser.add_argument("-verbose", default=False, action='store_true')
    parser.add_argument("-output", help='temporary output file (.pickle)')
    args = parser.parse_args()

    if args.start_time:
        st = args.start_time
    else:
        st = -np.inf

    if args.end_time:
        et = args.end_time
    else:
        et = np.inf

    raw_hbonds = do_hbonds(args.traj, topology=args.top, verbose=args.verbose, distance_cutoff=args.cutoff,
                           angle_cutoff=args.cutoff_angle, start_time=st, end_time=et, skip=args.skip,
                           chunk_size=args.chunk_size, periodic=args.periodic, exclude_waters=args.exclude_waters,
                           sidechain_only=args.sidechain_only)

    pickle.dump(raw_hbonds, open(args.output, 'wb'))
