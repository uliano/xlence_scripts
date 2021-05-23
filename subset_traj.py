#!/usr/bin/env python

import argparse
import glob
from MDAnalysis import transformations, Universe, Writer


def main():
    parser = argparse.ArgumentParser(description='saves new topology and joined trajectory with only sel(ected) atoms.')
    parser.add_argument('-t', '--topology', help='the topology file (pdb, gro, psf...)')
    parser.add_argument('-sel', '--selection', help='the selection in MDA language')
    parser.add_argument('-o', '--output', help='output basename')
    parser.add_argument('-s', '--slice', help='slicing output trajectory START:END:SKIP')
    parser.add_argument('--reset_time', help='make trajectory start from time 0', action='store_true')
    parser.add_argument('trajectory', help='the trajectory(ies) file(s). Accepts globbing', nargs=argparse.REMAINDER)
    parser.add_argument('--dcd', help='writes traj in dcd format', action='store_true')
    parser.add_argument('-u', '--unwrap', action='store_true', help='unwrap PBC')
    args = parser.parse_args()

    # expand input trajectory names
    traj_files = []
    for traj_arg in args.trajectory:
        tf = [tfile for tfile in glob.glob(traj_arg)]
        traj_files += tf

    traj_files.sort()
    if args.slice:
    	slicer = slice(*[int(x) if x else None for x in args.slice.split(':')])
    else:
        slicer = slice(None, None, None)    

    u = Universe(args.topology, *traj_files)

    if args.unwrap:
        workflow = [transformations.unwrap(u.atoms)]
        u.trajectory.add_transformations(*workflow)

    if args.reset_time:
        time_offset = u.trajectory[0].time
    else:
        time_offset = 0.0

    if args.selection:
        selection = u.select_atoms(args.selection)
    else:
        selection = u.atoms



    selection.write(f'{args.output}.pdb')

    if args.dcd:
        traj_file=f'{args.output}.dcd'
    else:
        traj_file=f'{args.output}.xtc'
    with Writer(traj_file, selection.n_atoms) as write_handle:
        for time_frame in u.trajectory[slicer]:
            time_frame.time -= time_offset
            write_handle.write(selection)


if __name__ == '__main__':
    main()
