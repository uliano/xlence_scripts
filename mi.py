#!/usr/bin/env python

import argparse
import multiprocessing as mp
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors
from tqdm import tqdm

DEFAULT_CA = "default_CA"
SCHR_CA = "a.pt CA"
MDA_CA = "name CA"

# ERROR: how to pass k to calculate_mi ??

def _worker(k):
    def inner(pair):
        return calculate_mi(*pair, k=k)

    return inner

def worker(pair):
    return calculate_mi(*pair, k=6)


def get_MI(x, k=6, njobs=1, dump=False):
    nt, na, d = x.shape
    x = x - x.mean(axis=0)  # get atom fluctuations oround their mean position
    # normalize fluctuation interval between 0, 1 WARNING: is this required?
    x = (x - x.min(axis=0)) / np.ptp(x, axis=0)
    a = x.transpose((1, 0, -1))  # -> (na, nt, d)

    pairs = ((a[i], a[j]) for i in range(na) for j in range(i))

    MI = np.zeros((na, na))

    start_time = time.time()
    with mp.Pool(processes=njobs) as pool:
        res = pool.imap(worker, pairs)
        # MI[np.tril_indices_from(MI, -1)] = np.fromiter(res, dtype=np.float64)
        MI[np.tril_indices_from(MI, -1)] = np.fromiter(
            tqdm(res, total=int(na * (na - 1) / 2)),
            dtype=np.float64
        )
    print()
    print(
        f"finished {na} atoms for {nt} frames in {timedelta(seconds=time.time() - start_time)}"
    )

    MI += np.flip(MI)

    if dump:
        import pickle

        with open(dump + ".pickle", "wb") as f:
            pickle.dump(MI, f)

    # postprocess
    # eq. 9 from https://www.mpibpc.mpg.de/276284/paper_generalized_corr_lange.pdf
    MI = np.sqrt(1 - np.exp(-2 * MI))
    MI[np.diag_indices_from(MI)] = 1
    return MI


def calculate_mi(x, y, k=6):
    """
    Calculate Mutual Information between two continuous multidimensional random variables
    using Nearest Neighbour approximation for entropy as described in [1].

    :param x, y: np.ndarray of shape (n_samples, n_dim)
    :kwarg    k: number of neighbors to use to estimate entropy
    :return: scalar MI score I(x;y)

    References
    ----------
    [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
        information". Phys. Rev. E 69, 2004.
    """
    # x, y = (n, d)
    # metric = DistanceMetric.get_metric('euclidean')
    # xx = metric.pairwise(x)
    # yy = metric.pairwise(y)
    # zz = np.maximum(xx, yy)
    # zz.sort(axis=1)
    # radi = np.nextafter(zz[:, k], 0)

    xy = np.hstack((x, y))  # (n, 2*d)
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k)
    nn.fit(xy)
    radi = nn.kneighbors()[0]
    radi = np.nextafter(radi[:, -1], 0)

    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radi, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radi, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    n_samples = len(xy)

    mi = (
        digamma(n_samples)
        + digamma(k)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def get_cormat(x):
    """
    Get covariance matrix of atom fluctuations

    :param x: atom positions over time, np.ndarray of shape (n_time, n_atoms, n_dims)
    :return: the normalized covariance matrix, np.ndarray of shape (n_atoms, n_atoms)
    """

    fluct = x - np.mean(x, axis=0)
    dots = 1 / x.shape[0] * np.einsum("ijk,ilk->jl", fluct, fluct)

    diagonal = np.diag(dots)
    norm_matrix = np.outer(diagonal, diagonal)
    norm_matrix = np.sqrt(np.absolute(norm_matrix))
    corr_matrix = np.divide(dots, norm_matrix)
    return corr_matrix


def preproc_schrodinger(args):
    from schrodinger.application.desmond.packages import topo, traj, traj_util

    align_sel = SCHR_CA if args.align == DEFAULT_CA else args.align
    corr_sel = [SCHR_CA] if args.asl == DEFAULT_CA else args.asl

    if args.t:
        msys, cms = topo.read_cms(args.cms)
        trj = traj.read_traj(args.t[0])
    else:
        msys, cms, trj = traj_util.read_cms_and_traj(args.cms)

    slicer = (
        slice(*[int(i) if i else None for i in args.s.split(":")])
        if args.s
        else slice(None, None)
    )

    trj = trj[slicer]

    if args.align:
        print(f'Aligning trajectory to "{align_sel}"')
        fit_gids = topo.asl2gids(cms, align_sel)
        ref = cms.extract(cms.select_atom(align_sel)).getXYZ()
        trj = topo.superimpose(msys, fit_gids, trj, ref)

    x = []
    for sel in corr_sel:
        gids = topo.asl2gids(cms, sel)
        x.append(np.array([fr.pos(gids) for fr in trj]))  # type: ignore
    x = np.concatenate(x, axis=1)
    return x


def preproc_mda(args):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    align_sel = MDA_CA if args.align == DEFAULT_CA else args.align
    corr_sel = [MDA_CA] if args.asl == DEFAULT_CA else args.asl

    top = args.cms
    slicer = (
        slice(*[int(i) if i else None for i in args.s.split(":")])
        if args.s
        else slice(None, None)
    )

    U = mda.Universe(top, *args.t)

    if args.align:
        print(f'Aligning trajectory to "{align_sel}"')
        U.trajectory[0]
        mobile = U.copy()
        align.AlignTraj(mobile, U, select=align_sel, in_memory=True).run()
        U = mobile

    x = []
    for sel in corr_sel:
        U.trajectory[0]
        atoms = U.select_atoms(sel)
        x.append(np.array([atoms.positions for _ in U.trajectory[slicer]]))
    x = np.concatenate(x, axis=1)
    return x


def main():
    parser = argparse.ArgumentParser(
        description="""
    Calculate generalized correlations based on mutual information of selected atoms fluctuations.
    example:

    mi.py my_md.pdb corr_out -t my_md.xtc -backend mda -j 16 -s ::10 -align "name CA and segid A" -asl "name CA and segid A" "name CB and segid B" -corr
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "cms",
        help="input topology file: cms for schrodinger backend, or any format supported by MDAnalysis",
    )
    parser.add_argument("out", help="base name for the output")
    parser.add_argument(
        "-backend",
        help="preprocess trajectory using selected backend: MDAanalysis (mda) or Schrodinger (schr)",
        choices=["mda", "schr"],
        default="schr",
    )
    parser.add_argument(
        "-t",
        help="trj dir. If backend is MDAnalysis, multiple trajectories to be concatenated can be supplied",
        metavar="trajectory",
        nargs="+",
    )
    parser.add_argument(
        "-j",
        help="number of processes (max physical cores), default 1",
        type=int,
        metavar="N cores",
        default=1,
    )
    parser.add_argument(
        "-asl",
        help="atom selection. You can specify multiple selections that will be concatenated sequentially and will appear as contiguous blocks in the results. Default CA",
        nargs="+",
        default=DEFAULT_CA,
    )
    parser.add_argument("-s", help="slicer", metavar="START:END:STEP")
    parser.add_argument(
        "-pickle", help="dump unmodified MI matrix to pickle file", action="store_true"
    )
    parser.add_argument(
        "-corr", help="calculate the correlation matrix", action="store_true"
    )
    parser.add_argument(
        "-align",
        help="asl to align trajectory, default CA",
        const=DEFAULT_CA,
        nargs="?",
        metavar="ASL",
    )
    parser.add_argument(
        "-k", help="number of neighbors to use to estimane entropy", type=int
    )
    parser.add_argument(
        "-metric",
        help="scikit-learn distance metric to estimate distances between fluctuations",
    )
    args = parser.parse_args()

    if args.backend == "schr":
        preproc = preproc_schrodinger
    else:
        preproc = preproc_mda

    x = preproc(args)
    nt, na, d = x.shape

    print("Calculating Generalized Correlations ...")
    MI = get_MI(x, k=args.k, njobs=args.j, dump=args.pickle and args.out)
    print("Done.")

    with open(args.out + "_MI.dat", "w") as f:
        f.write(f"# MI matrix, {na} x {na}\n")
        for row in MI:
            f.write(" ".join(str(i) for i in row))
            f.write("\n")

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle("Mutual Information")
    plt.imshow(MI, origin="lower", cmap="inferno")
    plt.colorbar()
    fig.tight_layout()
    plt.savefig(args.out + "_MI.png")

    if args.corr:
        print("Calculating Pearson Correlation Coefficients Matrix")
        C = get_cormat(x.reshape(nt, na, d))
        print("Done.")
        with open(args.out + "_Cor.dat", "w") as f:
            f.write(f"# Correlation matrix, {na} x {na}\n")
            for row in C:
                f.write(" ".join(str(i) for i in row))
                f.write("\n")
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle("Correlation Matrix")
        plt.imshow(C, origin="lower", cmap="inferno")
        plt.colorbar()
        fig.tight_layout()
        plt.savefig(args.out + "_Cor.png")

        fig = plt.figure(figsize=(15, 15))
        fig.suptitle("MI/Cor Matrix")
        plt.imshow(np.tril(MI) + np.triu(C, k=1), origin="lower", cmap="inferno")
        plt.colorbar()
        fig.tight_layout()
        plt.savefig(args.out + "_MICor.png")

    print("All done.")
    print(
        r"""
              ____
            /____ `\
           ||_  _`\ \
     .-.   `|O, O  ||
     | |    (/    -)\
     | |    |`-'` |\`
  __/  |    | _/  |
 (___) \.  _.\__. `\___
 (___)  )\/  \    _/  ~\.
 (___) . \   `--  _   `\
  (__)-    ,/        (   |
       `--~|         |   |
           |         |   | ")
    """
    )


if __name__ == "__main__":
    main()
