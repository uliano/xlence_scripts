import argparse
import pickle
import json
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from scipy.stats import binned_statistic_dd, iqr, kstat, moment


def beta(T):
    # 8.314462618e-3 kJ/(mol K)
    # 1.987204259e-3 kcal/(mol K)
    return 1 / (0.001987204259 * T)


def exponential_average(a, dv, hist, T):
    def funct(dv_j):
        if not len(dv_j):
            return np.nan
        return np.mean(np.exp(beta(T) * dv_j.astype('float128')))

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
        default='maclaurin'
    )
    parser.add_argument(
        "-thresh", help="threshold for calculating cumulant expansion. Default=10"
    )
    parser.add_argument(
        "-bins",
        help="comma-separated number of bins along each dimension or a single number for all",
        default='30'
    )

    parser.add_argument(
        "-pickle",
        help="dump all data in pickle format",
        action='store_true'
    )

    parser.add_argument(
        "-json",
        help="dump all data in json format",
        action='store_true'
    )

    parser.add_argument(
        "-weights",
        help="plot weights",
        action='store_true'
    )

    parser.add_argument(
        "-histogram",
        help="plot histogram",
        action='store_true'
    )

    parser.add_argument(
        "-min",
        help="comma separated min for each dimension",
    )

    parser.add_argument(
        "-max",
        help="comma separated min for each dimension",
    )

    parser.add_argument(
        "-er",
        help="energy range for selecting frames. max (min=0 is then assumed) or min,max",
    )

    parser.add_argument(
        "-o", "--output",
        help="output prefix",
        default='reweighted'
    )
    args = parser.parse_args()

    data = np.genfromtxt(args.data)
    if len(data.shape) == 2:
        ndata, ndim = data.shape
    else:
        ndata = data.shape
        ndim=1
    with open(args.data, 'r') as f:
        line = f.readline()
        if line[0] == '#':
            cv = line[1:].split()
        else:
            cv = [f'variable{i+1}' for i in range(ndim)]

    dv = np.genfromtxt(args.dv)
    if dv.ndim == 2:
        dv = dv[:, -1]

    if ',' in args.bins:
        nbin = tuple([int(e) for e in args.bins.split(',')])
    else:
        nbin = tuple([int(args.bins) for _ in range(ndim)])

    if args.min:
        if ',' in args.min:
            mins = [int(e) for e in args.min.split(',')]
        else:
            mins = [int(args.min) for _ in range(ndim)]
    else:
        mins = data.min(axis=0)

    if args.max:
        if ',' in args.max:
            maxs = [int(e) for e in args.max.split(',')]
        else:
            maxs = [int(args.max) for _ in range(ndim)]
    else:
        maxs = data.max(axis=0)

    if args.er:
        if ',' in args.er:
            energy_range = [float(e) for e in args.er.split(',')]
        else:
            energy_range = [0.0, float(args.er)]
    else:
        energy_range = None

    return data, dv, ndata, ndim, nbin, cv, mins, maxs, args.output, args.k, args.t, args.method, energy_range, args


def plot_2d(edge1, edge2, cv1, cv2, array, title, filename):
    plt.figure()
    cmap = mpl.cm.viridis.copy()
    plt.contourf(edge2[:-1], edge1[:-1], array, cmap=cmap, origin="lower")
    plt.ylabel(cv1)
    plt.xlabel(cv2)
    plt.title(title)

    # plt.imshow(pmf.T, cmap=cmap, origin="lower", interpolation='gaussian')
    plt.colorbar()
    plt.savefig(filename, dpi=600)


def plot_1d(edge, cv, array, title, filename):
    plt.figure()
    plt.plot(edge[:-1], array, linestyle='-')
    plt.ylabel(cv)
    plt.title(title)
    plt.savefig(filename, dpi=600)

def main():
    data, dv, ndata, ndim, nbin, cv, mins, maxs, basename, order, temp, method, energy_range, args = parse_args()
    if ndim > 1:
        data_range = [r for r in zip(mins, maxs)]
    else:
        data_range = ((mins, maxs),)

    hist = binned_statistic_dd(data, dv, statistic=lambda dv: len(dv) or np.nan, bins=nbin , range=data_range)  # type: ignore
    bn = hist.binnumber
    binnr_fix = []
    for bn in hist.binnumber:
        indices = np.unravel_index(bn, np.array(nbin) + 2, order='C')
        if 0 in indices or np.prod(np.array(nbin) + 2) in indices:
            binnr_fix.append(-1)
            continue
        indices = [axis - 1 for axis in indices]
        binnr_fix.append(np.ravel_multi_index(indices, nbin))

    pb = hist.statistic / np.nansum(hist.statistic)  # count to density, biased probability

    if method == "exp":
        weights = exponential_average(data, dv, hist, temp)
        p = pb * weights
        pmf = -1 / beta(temp) * np.log(p / np.nansum(p))
    elif method == "maclaurin":
        kwargs = dict(k=order) if order else {}
        weights = maclaurin_series(data, dv, hist, temp, **kwargs)
        p = pb * weights
        pmf = -1 / beta(temp) * np.log(p / np.nansum(p))
    elif method == "cumulant":
        kwargs = dict(k=order) if order else {}
        weights = cumulant_expansion(data, dv, hist, temp, **kwargs)
        p = np.log(pb) + weights # log(p) = log(pb) + log(weights)
        pmf = -1 / beta(temp) * p
    else:
        print("Unknown method. Exiting.")
        return

    pmf[np.isnan(pmf)] = np.nanmax(pmf)
    pmf -= np.nanmin(pmf)

    if energy_range:
        seeked_bins = np.nonzero((pmf >= energy_range[0]) & (pmf <= energy_range[1]))
        raveled_bins = np.ravel_multi_index(seeked_bins, nbin)
        frames = []
        for bin in raveled_bins:
            frames.extend(np.nonzero(binnr_fix == bin)[0])
        with open(basename + '_frames.dat', 'w') as fp:
            for frame in frames:
                fp.write(f'{frame}\n')

    # frames = np.nonzero(hist.binnumber == min_bin)

    cmap = mpl.cm.viridis.copy()
    # cmap.set_bad("white")
    edges = hist.bin_edges

    ind = np.unravel_index(np.argmin(pmf, axis=None), pmf.shape)

    if ndim > 2:
        for first, second in itertools.combinations(range(ndim),2):
            reduce_dims = set(range(ndim)) - set((first, second))
            pmf_temp = pmf.min(tuple(reduce_dims))
            plot_2d(edges[first], edges[second], cv[first], cv[second],pmf_temp ,
                    f"{basename} pmf",
                    basename + f'_{cv[first]}_{cv[second]}_pmf.png')
            if args.weights:
                weights_temp = weights.min(tuple(reduce_dims))
                plot_2d(edges[first], edges[second], cv[first], cv[second], weights_temp,
                        f"{basename} weights",
                        basename + f'_{cv[first]}_{cv[second]}_weights.png')
            if args.histogram:
                hist_temp = hist.statistic.min(tuple(reduce_dims))
                plot_2d(edges[first], edges[second], cv[first], cv[second], hist_temp,
                        f"{basename} histogram",
                        basename + f'_{cv[first]}_{cv[second]}_histogram.png')
    elif ndim == 2:
        plot_2d(edges[0], edges[1], cv[0], cv[1], pmf, f"{basename} pmf", basename + '_pmf.png')
        if args.weights:
            plot_2d(edges[0], edges[1], cv[0], cv[1], weights, f"{basename} weights", basename + '_weights.png')
        if args.histogram:
            plot_2d(edges[0], edges[1], cv[0], cv[1], hist.statistic, f"{basename} histogram", basename + '_histogram.png')
    else:
        plot_1d(edges[0], cv[0], pmf, f"{basename} pmf", basename + '_pmf.png')
        if args.weights:
            plot_1d(edges[0], cv[0], weights, f"{basename} weights", basename + '_weights.png')
        if args.histogram:
            plot_1d(edges[0], cv[0], hist.statistic, f"{basename} histogram", basename + '_histogram.png')


    if args.pickle:
        result = {'pmf': pmf, 'edges': edges,  'hist': hist.statistic, 'weights': weights, 'binnumber': hist.binnumber}
        with open(basename + '.pickle', 'wb') as fp:
            pickle.dump(result, fp)
    if args.json:
        result1 = {'pmf': pmf.tolist(), 'edges': [e.tolist() for e in edges],  'hist': hist.statistic.tolist(),
                   'weights': weights.tolist(), 'binnumber': hist.binnumber.tolist()}
        with open(basename + '.json', 'w') as fp:
            string = json.dumps(result1, indent=4, sort_keys=True)
            fp.write(string)



if __name__ == "__main__":
    main()
