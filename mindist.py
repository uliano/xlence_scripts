#!/usr/bin/env python
import argparse
from itertools import product
from tqdm import tqdm
from MDAnalysis import transformations, Universe
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def get_shortest_distance(coords1, coords2, cutoff=np.inf):
    """
    Determines the shortest distance and indices of the nearest atoms
    between two structures or between a groups of atoms in a single
    structure.
    NOTE: Periodic boundary conditions (PBC) are NOT honored.
    """
    kd_tree = cKDTree(coords1)
    distances, nn_indices = kd_tree.query(coords2, distance_upper_bound=cutoff)
    min_distance = min(distances)
    if min_distance == np.inf:
        # None of the distances where within the cutoff:
        return cutoff
    return min_distance


def main():
    parser = argparse.ArgumentParser(description='''Computes the minimum distance between\n
    2 selections (sel and sel2) or
    the same selection (sel) and its images in neighbouring periodic images.''')
    parser.add_argument('-t', dest='top', help='topology file')
    parser.add_argument('traj', help='trajectory file')
    parser.add_argument('-sel', help='atom selection')
    parser.add_argument('-sel2', help='atom selection2, ignored in periodic mode.')
    parser.add_argument('-p', '--periodic', action='store_true',
                        help='compute distance of "sel" between contiguous periodic cells')
    parser.add_argument('-u', '--unwrap', action='store_true', help='unwrap PBC')
    parser.add_argument('-o', '--output', help='output file basename')
    parser.add_argument('-c', '--cutoff', type=float, default=np.inf,
                        help='compute distances only for atoms closer than cutoff')
    parser.add_argument('-s', '--slice', default="::", help='slice trajectory, START:END:STEP')
    parser.add_argument('-f', '--fast', action='store_true',
                        help='in periodic mode consider only 6 neighboring images, otherwise ignored.')
    parser.add_argument('--print', action='store_true',
                        help='print mindist on stdout.')
    args = parser.parse_args()

    top = args.top  # topologia va bene anche pdb
    traj = args.traj  # traiettoria (.dcd, .xtc, .nc, ...)
    slicer = slice(*[int(x) if x else None for x in args.slice.split(':')])
    u = Universe(top, traj)
    if args.unwrap:
        workflow = [transformations.unwrap(u.atoms)]
        u.trajectory.add_transformations(*workflow)
    if not args.sel:
        p=u.atoms
    else:
        p = u.select_atoms(args.sel)

    min_dist = []

    if args.periodic:
        if args.fast:
            cells = [(1, 0, 0),
                     (-1, 0, 0),
                     (0, 1, 0),
                     (0, -1, 0),
                     (0, 0, 1),
                     (0, 0, -1)]
        else:
            i = (0, 1, -1)  # --> (0, 0, 0), (0, 0, 1) ...
            cells = np.array(list(product(i, i, i)))
            cells = cells[1:]
        for frame in tqdm(u.trajectory[slicer]):
            distances = []
            pos = p.positions
            if args.periodic:
                box = frame.dimensions[0:3]
                images = (pos + img for img in cells * box)
                for image in images:
                    distances.append((get_shortest_distance(pos, image, cutoff=args.cutoff), u.trajectory.time))
                    # get_shortest_distance -> (dist, anum1, anum2)
                min_dist.append(min(distances))
    else:
        p2 = u.select_atoms(args.sel2)
        for _ in tqdm(u.trajectory[slicer]):
            pos = p.positions
            pos2 = p2.positions
            min_dist.append((get_shortest_distance(pos, pos2, cutoff=args.cutoff), u.trajectory.time))

    min_dist = np.array(min_dist)

    with open(args.output + '.csv', 'w') as fh:
        fh.write('time,distance\n')
        for dist in min_dist:
            fh.write(f'{dist[1]},{dist[0]}\n')
    if max(min_dist[:, 1]) > 10000:
        plt.plot(min_dist[:, 1]/1000.0, min_dist[:, 0])
        plt.xlabel('time (ns)')
    else:
        plt.plot(min_dist[:, 1], min_dist[:, 0])
        plt.xlabel('time (ps)')
    plt.ylabel('min distance (Å)')
    plt.savefig(args.output + '.png')

    if args.print:
        print(f'The minimum distance is {np.min(min_dist[:, 0])} A')


if __name__ == "__main__":
    main()
