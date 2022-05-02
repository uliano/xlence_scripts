import argparse

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
        return np.mean(np.exp(beta(T) * dv_j))

    weights = binned_statistic_dd(a, dv, statistic=funct, binned_statistic_result=hist)  # type: ignore
    return weights.statistic / np.nansum(weights.statistic)


def maclaurin_series(a, dv, hist, T, k=5):
    def funct(dv_j):
        if not len(dv_j):
            return np.nan
        return sum(
            (beta(T) ** k_) / factorial(k_) * (dv_j**k_).mean() for k_ in range(k)
        )

    weights = binned_statistic_dd(a, dv, statistic=funct, binned_statistic_result=hist)  # type: ignore
    return weights.statistic / np.nansum(weights.statistic)


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
    return weights.statistic  # - np.log(np.nansum(np.exp(weights.statistic)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="collective variables (N, D)")
    parser.add_argument(
        "dV", help="bias potential (N, ) but reads the last column anyway..."
    )
    parser.add_argument("out", help="output name")
    parser.add_argument(
        "-k",
        help="Maclaurin series kth expansion (default: 5) or kth cumulant expansion (default: 3)",
        type=int,
    )
    parser.add_argument("-t", help="system temperature", type=float, required=True)
    parser.add_argument(
        "-method",
        help="average method",
        choices=["maclaurin", "cumulant", "exp"],
        required=True,
    )
    parser.add_argument(
        "-thresh", help="threshold for calculating cumulant expansion. Default=10"
    )
    parser.add_argument(
        "-bins",
        help="comma-separated number of bins X,Y,...,D. Defaults to Freedman Diaconis Estimator",
    )
    args = parser.parse_args()

    data = np.genfromtxt(args.data)
    dv = np.genfromtxt(args.dV)[:, -1]

    nbins = (
        args.bins
        and [int(bin) for bin in args.bins.split(",")]
        or 2 * iqr(data, axis=0) / (data.shape[0] ** (1 / 3))
    )

    hist = binned_statistic_dd(data, dv, statistic=lambda dv: len(dv) or np.nan, bins=nbins)  # type: ignore
    pb = hist.statistic / np.nansum(hist.statistic)  # count to density

    if args.method == "exp":
        weights = exponential_average(data, dv, hist, args.t)
        pmf = -1 / beta(args.t) * np.log(pb * weights)
    elif args.method == "maclaurin":
        kwargs = dict(k=args.k) if args.k else {}
        weights = maclaurin_series(data, dv, hist, args.t, **kwargs)
        pmf = -1 / beta(args.t) * np.log(pb * weights)
    elif args.method == "cumulant":
        kwargs = dict(k=args.k) if args.k else {}
        weights = cumulant_expansion(data, dv, hist, args.t, **kwargs)
        pmf = -1 / beta(args.t) * (np.log(pb) + weights)
    else:
        print("Unknown method. Exiting.")
        return

    pmf[np.isnan(pmf)] = np.nanmax(pmf)
    pmf -= np.nanmin(pmf)

    cmap = mpl.cm.viridis.copy()
    # cmap.set_bad("white")
    cv1, cv2 = hist.bin_edges
    plt.contourf(cv1[:-1], cv2[:-1], pmf.T, cmap=cmap, origin="lower")
    # plt.imshow(pmf.T, cmap=cmap, origin="lower", interpolation='gaussian')
    plt.colorbar()
    plt.savefig(args.out + ".png", dpi=600)


if __name__ == "__main__":
    main()
