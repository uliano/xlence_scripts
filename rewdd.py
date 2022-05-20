#!/usr/bin/env python

import argparse
import itertools
import json
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from scipy.stats import binned_statistic_dd, iqr, kstat, moment

# TODO: weights and histogram should be plotted using imshow.


def beta(T):
    # 8.314462618e-3 kJ/(mol K)
    # 1.987204259e-3 kcal/(mol K)
    return 1 / (0.001987204259 * T)


def exponential_average(a, dv, hist, T):
    def funct(dv_j):
        if not len(dv_j):
            return np.nan
        return np.mean(np.exp(beta(T) * dv_j.astype("float128")))

    weights = binned_statistic_dd(a, dv, statistic=funct, binned_statistic_result=hist)  # type: ignore
    return weights.statistic


def maclaurin_series(a, dv, hist, T, k=5):
    def funct(dv_j):
        if not len(dv_j):
            return np.nan
        return sum(
            (beta(T) ** k_) / factorial(k_) * (dv_j**k_).mean() for k_ in range(k)
        )

    weights = binned_statistic_dd(a, dv, statistic=funct, binned_statistic_result=hist)  # type: ignore
    return weights.statistic


def cumulant_expansion(a, dv, hist, T, k=3):
    def funct(dv_j):
        if len(dv_j) <= 10:
            return np.nan
        exp = sum(
            (beta(T) ** k_) / factorial(k_) * kstat(dv_j, k_) for k_ in range(1, k + 1)
        )
        return exp

    # this is the logarithm of mean(exp(beta*dv_j))
    weights = binned_statistic_dd(a, dv, statistic=funct, binned_statistic_result=hist)  # type: ignore
    return weights.statistic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", help="collective variables (ndata, ndimensions)")
    parser.add_argument(
        "-dv", help="bias potential (ndata, ) but reads the last column anyway..."
    )
    parser.add_argument(
        "-k",
        help="Maclaurin series kth expansion (default: 5) or kth cumulant expansion (default: 3)",
        type=int,
    )
    parser.add_argument("-t", help="system temperature", type=float, default=300)
    parser.add_argument(
        "-method",
        help="average method",
        choices=["maclaurin", "cumulant", "exp"],
        default="maclaurin",
    )

    parser.add_argument(
        "-thresh", help="threshold for calculating cumulant expansion. Default=10"
    )

    parser.add_argument(
        "-bins",
        help="comma-separated number of bins along each dimension or a single number for all",
        type=lambda args: tuple(int(i) for i in args.split(",")),
    )

    parser.add_argument(
        "-pickle", help="dump all data in pickle format", action="store_true"
    )

    parser.add_argument(
        "-json", help="dump all data in json format", action="store_true"
    )

    parser.add_argument("-weights", help="plot weights", action="store_true")

    parser.add_argument("-histogram", help="plot histogram", action="store_true")

    parser.add_argument(
        "-min",
        help="comma separated min for each dimension",
        type=lambda args: tuple(float(i) for i in args.split(",")),
    )

    parser.add_argument(
        "-max",
        help="comma separated max for each dimension",
        type=lambda args: tuple(float(i) for i in args.split(",")),
    )

    parser.add_argument(
        "-er",
        help="energy range for selecting frames. max (min=0 is then assumed) or min,max",
        type=lambda args: tuple(float(i) for i in args.split(",")),
    )

    parser.add_argument("-o", "--output", help="output prefix", default="reweighted")
    args = parser.parse_args()

    return args


def plot_2d(edge1, edge2, cv1, cv2, data, title, filename):
    plt.figure()
    plt.contourf(edge2[:-1], edge1[:-1], data, cmap="viridis", origin="lower")
    plt.ylabel(cv1)
    plt.xlabel(cv2)
    plt.title(title)

    # plt.imshow(pmf.T, cmap=cmap, origin="lower", interpolation='gaussian')
    plt.colorbar()
    plt.savefig(filename, dpi=600)


def plot_1d(edge, cv, data, title, filename):
    plt.figure()
    plt.plot(edge[:-1], data, linestyle="-")
    plt.ylabel(cv)
    plt.title(title)
    plt.savefig(filename, dpi=600)


def plot_dd(edges, cvs, data, title, filename):
    ndim = len(edges)
    for first, second in itertools.combinations(range(ndim), 2):
        reduce_dims = tuple(set(range(ndim)) - set((first, second)))
        data_slice = data.nanmean(axis=reduce_dims)
        fname = f"{filename}_{cvs[first]}_{cvs[second]}_pmf.png"
        plot_2d(
            edges[first],
            edges[second],
            cvs[first],
            cvs[second],
            data_slice,
            title,
            fname,
        )


def calculate_pmf(data, dv, nbin, data_range, method, k, temp):
    hist = binned_statistic_dd(data, dv, statistic=lambda dv: len(dv) or np.nan, bins=nbin, range=data_range)  # type: ignore

    binnr_fix = []
    for bn in hist.binnumber:
        indices = np.array(np.unravel_index(bn, np.array(nbin) + 2))
        if np.all([0 < i <= j for i, j in zip(indices, nbin)]):
            binnr_fix.append(np.ravel_multi_index(indices - 1, nbin))  # type: ignore
        else:
            binnr_fix.append(np.nan)

    pb = hist.statistic / np.nansum(
        hist.statistic
    )  # count to density, biased probability

    if method == "exp":
        weights = exponential_average(data, dv, hist, temp)
        p = pb * weights
        pmf = -1 / beta(temp) * np.log(p / np.nansum(p))
    elif method == "maclaurin":
        kwargs = dict(k=k) if k else {}
        weights = maclaurin_series(data, dv, hist, temp, **kwargs)
        p = pb * weights
        pmf = -1 / beta(temp) * np.log(p / np.nansum(p))
    elif method == "cumulant":
        kwargs = dict(k=k) if k else {}
        weights = cumulant_expansion(data, dv, hist, temp, **kwargs)
        p = np.log(pb) + weights  # log(p) = log(pb) + log(weights)
        pmf = -1 / beta(temp) * p
    else:
        raise ValueError(f"Unknown method {method}")

    pmf[np.isnan(pmf)] = np.nanmax(pmf)
    pmf -= np.nanmin(pmf)

    return hist, binnr_fix, weights, pmf


def validate_arg(arg, name, ndim):
    if (arglen := len(arg)) not in (1, ndim):
        raise ValueError(
            f"Argument {name} length should be 1 or {ndim}, but was {arglen}"
        )


def main():
    args = parse_args()

    data = np.genfromtxt(args.data)
    ndata, ndim = len(data), data.ndim > 1 and data.shape[-1] or 1
    data = data.reshape(-1, ndim)  # data is always 2d

    basename = args.output

    with open(args.data, "r") as f:
        line = f.readline()
        if line.startswith("#"):
            cv = line[1:].split()
        else:
            cv = [f"cv{i+1}" for i in range(ndim)]

    dv = np.genfromtxt(args.dv)
    dv = dv[:, -1] if dv.ndim > 1 else dv

    if args.bins:
        validate_arg(args.bins, "bins", ndim)
        nbin = args.bins if len(args.bins) == ndim else tuple(list(args.bins) * ndim)
    else:
        nbin = np.ptp(data, axis=0) / (
            (2 * iqr(data, axis=0) / (data.shape[0] ** (1 / 3)))
        )
        nbin = tuple(nbin.astype(int))

    if args.min:
        validate_arg(args.min, "min", ndim)
        mins = args.mins if len(args.mins) == ndim else tuple(list(args.mins) * ndim)
    else:
        mins = tuple(data.min(axis=0))

    if args.max:
        validate_arg(args.max, "max", ndim)
        maxs = args.maxs if len(args.maxs) == ndim else tuple(list(args.maxs) * ndim)
    else:
        maxs = tuple(data.max(axis=0))

    if args.er:
        validate_arg(args.er, "er", 2)
        energy_range = args.er if len(args.er) > 1 else (0, args.er[0])
    else:
        energy_range = None

    data_range = tuple(r for r in zip(mins, maxs))

    hist, binnumber, weights, pmf = calculate_pmf(
        data, dv, nbin, data_range, args.method, args.k, args.t
    )

    if energy_range:
        seeked_bins = np.nonzero((pmf >= energy_range[0]) & (pmf <= energy_range[1]))
        raveled_bins = np.ravel_multi_index(seeked_bins, nbin)
        frames = []
        for bin in raveled_bins:
            frames.extend(np.nonzero(binnumber == bin)[0])
        with open(basename + "_frames.dat", "w") as fp:
            for frame in frames:
                fp.write(f"{frame}\n")

    edges = hist.bin_edges

    if ndim == 1:
        plotter = plot_1d
        plot_args = (edges[0], cv[0])
    elif ndim == 2:
        plotter = plot_2d
        plot_args = (*edges, *cv)
    else:
        plotter = plot_dd
        plot_args = (edges, cv)

    plotter(*plot_args, pmf, f"{basename} pmf", basename + "_pmf.png")  # type: ignore

    if args.weights:
        plotter(*plot_args, weights, f"{basename} weights", basename + "_weights.png")  # type: ignore

    if args.histogram:
        plotter(*plot_args, hist.statistic, f"{basename} histogram", basename + "_histogram.png")  # type: ignore

    if args.pickle:
        result = {
            "pmf": pmf,
            "edges": edges,
            "hist": hist.statistic,
            "weights": weights,
            "binnumber": binnumber,
        }
        with open(basename + ".pickle", "wb") as fp:
            pickle.dump(result, fp)

    if args.json:
        result1 = {
            "pmf": pmf.tolist(),
            "edges": [e.tolist() for e in edges],
            "hist": hist.statistic.tolist(),
            "weights": weights.tolist(),
            "binnumber": binnumber.tolist(),
        }
        with open(basename + ".json", "w") as fp:
            string = json.dumps(result1, indent=4, sort_keys=True)
            fp.write(string)


if __name__ == "__main__":
    main()
