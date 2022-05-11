import argparse
import pickle
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="collective variables (N, D)")
    parser.add_argument(
        "dV", help="bias potential (N, ) but reads the last column anyway..."
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
        help="comma-separated number of bins along X and Y or a single number for both",
        default='30'
    )

    parser.add_argument(
        "-range",
        help="comma separated xmin,xmax,ymin,ymax",
    )

    parser.add_argument(
        "-o","--output",
        help="output prefix",
        default='reweighted'
    )
    args = parser.parse_args()

    data = np.genfromtxt(args.data)
    dv = np.genfromtxt(args.dV)
    if dv.ndim == 2:
        dv = dv[:, -1]
    if ',' in args.bins:
        nbins = args.bins.split(',')
        nbins = [int(e) for e in nbins]

    else:
        nbins=int(args.bins)

    if args.range:
        the_range = args.range.split(',')
        the_range = [int(e) for e in the_range]
        the_range = [the_range[:2], the_range[2:]]
    else:
        the_range = [[data[:,0].min(), data[:,0].max()], [data[:,1].min(), data[:,1].max()]]

    hist = binned_statistic_dd(data, dv, statistic=lambda dv: len(dv) or np.nan, bins=nbins, range=the_range)  # type: ignore
    pb = hist.statistic / np.nansum(hist.statistic)  # count to density, biased probability

    if args.method == "exp":
        weights = exponential_average(data, dv, hist, args.t)
        p = pb * weights
        pmf = -1 / beta(args.t) * np.log(p / np.nansum(p))
    elif args.method == "maclaurin":
        kwargs = dict(k=args.k) if args.k else {}
        weights = maclaurin_series(data, dv, hist, args.t, **kwargs)
        p = pb * weights
        pmf = -1 / beta(args.t) * np.log(p / np.nansum(p))
    elif args.method == "cumulant":
        kwargs = dict(k=args.k) if args.k else {}
        weights = cumulant_expansion(data, dv, hist, args.t, **kwargs)
        p = np.log(pb) + weights # log(p) = log(pb) + log(weights)
        pmf = -1 / beta(args.t) * p
    else:
        print("Unknown method. Exiting.")
        return

    pmf[np.isnan(pmf)] = np.nanmax(pmf)
    pmf -= np.nanmin(pmf)
    min_bin = np.argmin(pmf)
    frames = np.nonzero(hist.binnumber == min_bin)

    cmap = mpl.cm.viridis.copy()
    # cmap.set_bad("white")
    cv1, cv2 = hist.bin_edges

    ind = np.unravel_index(np.argmin(pmf, axis=None), pmf.shape)

    t1 = [cv1[ind[0]], cv1[ind[0] + 1]]
    t2 = [cv2[ind[1]], cv2[ind[1] + 1]]


    plt.contourf(cv1[:-1], cv2[:-1], pmf.T, cmap=cmap, origin="lower")
    # plt.imshow(pmf.T, cmap=cmap, origin="lower", interpolation='gaussian')
    plt.colorbar()
    plt.savefig(args.output + ".png", dpi=600)

    result = {'edge1': cv1, 'edge2': cv2, 'pmf': pmf, 'min1': t1, 'min2': t2,  'frames_min': frames,
              'binnumber': hist.binnumber}

    with open(args.output + '.pickle', 'wb') as fp:
        pickle.dump(result, fp)



if __name__ == "__main__":
    main()
