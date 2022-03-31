#!/usr/bin/env python

import argparse
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


def anharm(the_data):
    #    print "Compute anharmonicity"
    var = np.var(the_data)
    the_hist, the_edges = np.histogram(the_data, 50, normed=True)
    the_hist = np.add(the_hist, 0.000000000000000001)  # so that distrib
    d_x = the_edges[1] - the_edges[0]
    s_1 = -1 * np.trapz(np.multiply(the_hist, np.log(the_hist)), dx=d_x)
    s_2 = 0.5 * np.log(2.00 * np.pi * np.exp(1.0) * var + 0.000000000000000001)
    the_alpha = s_2 - s_1
    if np.isinf(the_alpha):
        the_alpha = 100
    return the_alpha


def reweight_ce(input_data, hist_min, binsx, dx, binsy, dy, dv, temp):
    hist2d, newedges_x, newedges_y = np.histogram2d(input_data[:, 0], input_data[:, 1],
                                                    bins=(binsx, binsy), weights=None)

    hist_max = np.max(hist2d)

    beta = 1.0 / (0.001987 * temp)
    nf = len(input_data[:, 0])
    nbins_x = len(hist2d[:, 0])
    nbins_y = len(hist2d[0, :])

    c_1 = np.zeros((nbins_x, nbins_y))
    c_2 = np.zeros((nbins_x, nbins_y))
    c_3 = np.zeros((nbins_x, nbins_y))

    binf_x = np.zeros(nf)  # array for storing assigned bin of each frame
    binf_y = np.zeros(nf)  # array for storing assigned bin of each frame
    n_a = np.zeros((nbins_x, nbins_y), dtype=int)  # n_a is equivalent to hist here
    dv_avg = np.zeros((nbins_x, nbins_y))
    dv_avg2 = np.zeros((nbins_x, nbins_y))
    dv_avg3 = np.zeros((nbins_x, nbins_y))
    dv_std = np.zeros((nbins_x, nbins_y))
    dv_anharm = np.zeros((nbins_x, nbins_y))
    dv_mat = np.zeros((nbins_x, nbins_y, int(hist_max)))  # matrix for storing dV of each assigned

    dv_avg_all = np.average(dv)
    dv_std_all = np.std(dv)
    # print('dV all: avg = ', dv_avg_all, 'std = ', dv_std_all)

    diff_tol_avg = 10
    diff_tol_std = 1

    for i in range(len(input_data[:, 0])):
        j_x = int((input_data[i, 0] - binsx[0]) / dx)
        j_y = int((input_data[i, 1] - binsy[0]) / dy)
        if j_x > nbins_x:
            j_x = nbins_x - 1
        if j_y > nbins_y:
            j_y = nbins_y - 1
        binf_x[i] = j_x
        binf_y[i] = j_y
        dv_mat[j_x, j_y, n_a[j_x, j_y]] = dv[i]
        n_a[j_x, j_y] = n_a[j_x, j_y] + 1

    for j_x in range(nbins_x):
        for j_y in range(nbins_y):
            dv_anharm[j_x, j_y] = 100
            if n_a[j_x, j_y] >= hist_min:
                num = int(n_a[j_x, j_y])
                atemp = np.zeros(num)
                atemp2 = np.zeros(num)
                atemp3 = np.zeros(num)
                for kk in range(num):
                    atemp[kk] = dv_mat[j_x, j_y, kk]
                    atemp2[kk] = dv_mat[j_x, j_y, kk] ** 2
                    atemp3[kk] = dv_mat[j_x, j_y, kk] ** 3
                dv_avg[j_x, j_y] = np.average(atemp)
                dv_std[j_x, j_y] = np.std(atemp)
                dv_anharm[j_x, j_y] = anharm(atemp)

                if np.absolute(dv_avg[j_x, j_y] - dv_avg_all) > diff_tol_avg or np.absolute(
                        dv_std[j_x, j_y] - dv_std_all) > diff_tol_std:
                    dv_avg[j_x, j_y] = 0
                    dv_std[j_x, j_y] = 0

                dv_avg2[j_x, j_y] = np.average(atemp2)
                dv_avg3[j_x, j_y] = np.average(atemp3)
                del atemp
                del atemp2
                del atemp3
                c_1[j_x, j_y] = beta * dv_avg[j_x, j_y]
                c_2[j_x, j_y] = 0.5 * beta ** 2 * dv_std[j_x, j_y] ** 2
                c_3[j_x, j_y] = (1.0 / 6.0) * beta ** 3 * (dv_avg3[j_x, j_y] - 3.0 * dv_avg2[j_x, j_y] *
                                                           dv_avg[j_x, j_y] + 2.0 * dv_avg[j_x, j_y] ** 3)
    return hist2d, newedges_x, newedges_y, c_1, c_2, c_3


def reweight_dv(input_data, hist_min, binsx, binsy, dx, dy, dv):
    hist2d, newedges_x, newedges_y = np.histogram2d(input_data[:, 0], input_data[:, 1],
                                                    bins=(binsx, binsy), weights=None)
    hist_max = np.max(hist2d)
    #   print np.max(hist2d)

    nf = len(input_data[:, 0])
    nbins_x = len(hist2d[:, 0])
    nbins_y = len(hist2d[0, :])

    binf_x = np.zeros(nf)  # array for storing assigned bin of each frame
    binf_y = np.zeros(nf)  # array for storing assigned bin of each frame
    n_a = np.zeros((nbins_x, nbins_y))  # n_a is equivalent to hist here
    dv_avg = np.zeros((nbins_x, nbins_y))
    dv_std = np.zeros((nbins_x, nbins_y))
    dv_anharm = np.zeros((nbins_x, nbins_y))
    dv_mat = np.zeros((nbins_x, nbins_y, hist_max))  # matrix for storing dV of each assigned

    for i in range(len(input_data[:, 0])):
        j_x = int((input_data[i, 0] - binsx[0]) / dx)
        j_y = int((input_data[i, 1] - binsy[0]) / dy)
        if j_x > nbins_x:
            j_x = nbins_x - 1
        if j_y > nbins_y:
            j_y = nbins_y - 1
        binf_x[i] = j_x
        binf_y[i] = j_y
        dv_mat[j_x, j_y, n_a[j_x, j_y]] = dv[i]
        n_a[j_x, j_y] = n_a[j_x, j_y] + 1

    for j_x in range(nbins_x):
        for j_y in range(nbins_y):
            dv_anharm[j_x, j_y] = 100
            if n_a[j_x, j_y] >= hist_min:
                num = int(n_a[j_x, j_y])
                atemp = np.zeros(num)
                for kk in range(num):
                    atemp[kk] = dv_mat[j_x, j_y, kk]
                dv_avg[j_x, j_y] = np.average(atemp)
                dv_std[j_x, j_y] = np.std(atemp)
                dv_anharm[j_x, j_y] = anharm(atemp)
                del atemp
    return hist2d, newedges_x, newedges_y, binf_x, binf_y, dv_avg, dv_std, dv_anharm, dv_mat


def hist2pmf2d(the_hist, hist_min, temp):
    nbins_x = len(the_hist[:, 0])
    nbins_y = len(the_hist[0, :])
    p_m_f = np.zeros((nbins_x, nbins_y))
    pmf_min = 100
    for j_x in range(len(the_hist[:, 0])):
        for j_y in range(len(the_hist[0, :])):
            if the_hist[j_x, j_y] >= hist_min:
                p_m_f[j_x, j_y] = -(0.001987 * temp) * np.log(the_hist[j_x, j_y])
            if pmf_min > p_m_f[j_x, j_y]:
                pmf_min = p_m_f[j_x, j_y]
    #        p_m_f=p_m_f-pmf_min  ## zero value to lowest energy state
    return p_m_f


def normalize2d(p_m_f, emax):
    p_m_f = p_m_f - np.min(p_m_f)  # zero value to lowest energy state
    temphist = p_m_f
    # print "rows = ", len(temphist[:,0])
    # print "cols = ", len(temphist[0,:])
    # set infinity free energy values to is cb_max
    for j_y in range(len(temphist[0, :])):
        for j_x in range(len(temphist[:, 0])):
            if np.isinf(temphist[j_x, j_y]):
                temphist[j_x, j_y] = emax
    return temphist


def prephist(hist_2, temp, cb_max):
    hist_2 = np.add(hist_2, 0.000000000000000001)  # so that distrib
    hist_2 = (0.001987 * temp) * np.log(hist_2)  # Convert to free energy in Kcal/mol
    hist_2 = np.max(hist_2) - hist_2  # zero value to lowest energy state
    #   print np.max(hist2)
    temphist2 = hist_2
    # set infinity free energy values to is cb_max
    for j_y in range(len(temphist2[0, :])):
        for j_x in range(len(temphist2[:, 0])):
            if np.isinf(temphist2[j_x, j_y]):
                temphist2[j_x, j_y] = cb_max
    return temphist2


def output_pmf2d(the_file, the_hist, binsx, binsy):
    fpmf = open(the_file, 'w')
    strpmf = '#RC1\tRC2\tPMF(kcal/mol)\n\n@    xaxis  label \"RC1\"\n@    yaxis  label \"RC2\"\n@TYPE xy\n'
    fpmf.write(strpmf)
    for j_x in range(len(the_hist[:, 0])):
        for j_y in range(len(the_hist[0, :])):
            strpmf = str(binsx[j_x]) + ' \t' + str(binsy[j_y]) + ' \t' + str(the_hist[j_x, j_y]) + '\n'
            fpmf.write(strpmf)
    return fpmf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument("-i", "--input", required=True, help="2D input file")
    parser.add_argument("-j", "--job", required=True, choices=['noweight', 'weighthist', 'amd_time', 'amd_dV',
                                                               'amdweight', 'amdweight_MC', 'amdweight_CE'])
    parser.add_argument("-w", "--weight", required=True, help="weight file")
    parser.add_argument("-xmin", type=float)
    parser.add_argument("-xmax", type=float)
    parser.add_argument("-ymin", type=float)
    parser.add_argument("-ymax", type=float)
    parser.add_argument("-dx", type=float, default=6.0, help="Discretization size in X dimension")
    parser.add_argument("-dy", type=float, default=6.0, help="Discretization size in Y dimension")
    parser.add_argument("-cutoff", type=float, default=10.0, help="histogram cutoff")
    parser.add_argument("-T", dest="T", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("-Emax", dest="Emax", type=float, default=8.0, help="Maximum free energy")
    parser.add_argument("-order", dest="order", type=int, default=10, help="Order of Maclaurin series")
    args = parser.parse_args()

    inputfile = np.loadtxt(args.input)
    rows = inputfile.shape[0]

    if args.job == "weighthist":
        data = np.loadtxt(args.weight)
        weights = data[:, 0]
        dV = np.zeros(rows)
    elif args.job in ["amd_time", "amd_dV", "amdweight", "amdweight_MC", "amdweight_CE"]:
        data = np.loadtxt(args.weight)
        weights = np.exp(data[:, 0])
        dV = data[:, 2]
    elif args.job == "noweight":
        weights = np.zeros(rows)
        weights = weights + 1
        dV = np.zeros(rows)
    else:
        raise ValueError("Unrecognized Job:", args.job)

    if args.xmin:
        xmin = args.xmin
    else:
        xmin = np.floor(np.amin(inputfile[:, 0]))
    if args.xmax:
        xmax = args.xmax
    else:
        xmax = np.ceil(np.amax(inputfile[:, 0]))
    dX = args.dx
    xbins = np.arange(xmin, (xmax + dX), dX)

    if args.ymin:
        ymin = args.ymin
    else:
        ymin = np.floor(np.amin(inputfile[:, 1]))
    if args.ymax:
        ymax = args.ymax
    else:
        ymax = np.ceil(np.amax(inputfile[:, 1]))
    dY = args.dy
    ybins = np.arange(ymin, (ymax + dY), dY)

    cutoff = args.cutoff
    Emax = args.Emax
    order = args.order
    T = args.T
    Beta = 1.0 / (0.001987 * T)

    plt_figs = True

    # reweighting
    if args.job == "amdweight_CE":
        hist2, newedgesX, newedgesY, c1, c2, c3 = reweight_ce(inputfile, cutoff, xbins, dX, ybins, dY, dV, T)
        pmf = hist2pmf2d(hist2, cutoff, T)
        c1 = -np.multiply(1.0 / Beta, c1)
        c2 = -np.multiply(1.0 / Beta, c2)
        c3 = -np.multiply(1.0 / Beta, c3)

        c12 = np.add(c1, c2)
        c123 = np.add(c12, c3)
        pmf_c1 = np.add(pmf, c1)
        pmf_c1 = normalize2d(pmf_c1, Emax)
        output_pmf2d('pmf-c1-' + str(args.input) + '.xvg', pmf_c1, xbins, ybins)

        pmf_c2 = np.add(pmf, c12)
        pmf_c2 = normalize2d(pmf_c2, Emax)
        output_pmf2d('pmf-c2-' + str(args.input) + '.xvg', pmf_c2, xbins, ybins)

        pmf_c3 = np.add(pmf, c123)
        pmf_c3 = normalize2d(pmf_c3, Emax)
        output_pmf2d('pmf-c3-' + str(args.input) + '.xvg', pmf_c3, xbins, ybins)

    elif args.job == "amdweight_MC":
        n = order
        MCweight = np.zeros(len(dV))
        beta_dV = np.multiply(dV, Beta)
        for x in range(0, n + 1):
            MCweight = np.add(MCweight, (np.divide(np.power(beta_dV, x), float(factorial(x)))))
        weights = MCweight
        hist2, newedgesX, newedgesY = np.histogram2d(inputfile[:, 0], inputfile[:, 1], bins=(xbins, ybins),
                                                     weights=weights)
        hist2 = prephist(hist2, T, Emax)
        output_pmf2d('pmf-' + str(args.input) + '.xvg', hist2, xbins, ybins)
    elif args.job == "amdweight":
        hist2, newedgesX, newedgesY = np.histogram2d(inputfile[:, 0], inputfile[:, 1], bins=(xbins, ybins),
                                                     weights=weights)
        hist2 = prephist(hist2, T, Emax)
        output_pmf2d('pmf-' + str(args.input) + '.xvg', hist2, xbins, ybins)
    else:
        hist2, newedgesX, newedgesY = np.histogram2d(inputfile[:, 0], inputfile[:, 1], bins=(xbins, ybins),
                                                     weights=None)
        hist2 = prephist(hist2, T, Emax)

    if args.job == "noweight":
        output_pmf2d('pmf-' + str(args.input) + '.xvg', hist2, xbins, ybins)

    if args.job == "amd_dV":
        plt_figs = False
        hist2, newedgesX, newedgesY, binfX, binfY, dV_avg, dV_std, dV_anharm, dV_mat = reweight_dv(inputfile, cutoff,
                                                                                                   xbins, ybins, dX,
                                                                                                   dY, dV)

        with open('dV-hist-2D-' + str(args.input) + '.xvg', 'w') as pmf_file:
            pmf_str = '#dV \tp(dV) \n\n@    xaxis  label \"dV\"\n@    yaxis  label \"p(dV)\"\n@TYPE xy\n'
            hist_dV, bin_dV = np.histogram(dV, bins=50)
            for k in range(len(hist_dV)):
                pmf_str = pmf_str + str(bin_dV[k]) + ' \t' + str(hist_dV[k]) + ' \n'
            pmf_file.write(pmf_str)

            for jx in range(len(hist2[:, 0])):
                for jy in range(len(hist2[0, :])):
                    nf_j = int(hist2[jx, jy])
                    if nf_j > 0:
                        pmffile = 'dV-hist-2D-' + str(args.input) + '-RCX' + str('%#08.2f' % xbins[jx]) + '-RCY' + str(
                            '%#08.2f' % ybins[jy]) + '.xvg'

        alpha = anharm(dV)
        # print("Anharmonicity of all dV = " + str(alpha))

        with open('dV-anharm-2D-' + str(args.input) + '.xvg', 'w') as pmf_file:
            pmf_str = '#RC \tdV_anharm \tError\n\n@    xaxis  label \"RC\"\n@    yaxis  label \"dV_anmarm\"\n@TYPE xy\n'
            pmf_file.write(pmf_str)
            for jx in range(len(dV_anharm[:, 0])):
                for jy in range(len(dV_anharm[0, :])):
                    pmf_str = str(xbins[jx]) + ' \t' + str(ybins[jy]) + ' \t' + str(dV_anharm[jx, jy]) + '\n'
                    pmf_file.write(pmf_str)

        with open('dV-stat-2D-' + str(args.input) + '.xvg', 'w') as pmf_file:
            pmf_str = '#RC \tdV_avg(kcal/mol) \tError\n\n@    xaxis  label \"RC\"\n'\
                     '@    yaxis  label \"dV(kcal/mol)\"\n@TYPE xydy\n'
            pmf_file.write(pmf_str)
            for jx in range(len(dV_anharm[:, 0])):
                for jy in range(len(dV_anharm[0, :])):
                    pmf_str = str(xbins[jx]) + ' \t' + str(ybins[jy]) + ' \t' + str(dV_avg[jx, jy]) + ' \t' + str(
                        dV_std[jx, jy]) + ' \t' + str(dV_anharm[jx, jy]) + '\n'
                    pmf_file.write(pmf_str)

        with open('dV-mat-2D-' + str(args.input) + '.xvg', 'w') as pmf_file:
            pmf_str = '#RC \tNf \tdV_avg \tdV_std \tdV_ij \n\n@    xaxis  '\
                     'label \"RC\"\n@    yaxis  label \"dV(kcal/mol)\"\n@TYPE xy\n'
            pmf_file.write(pmf_str)
            for jx in range(len(hist2[:, 0])):
                for jy in range(len(hist2[0, :])):
                    nf_j = int(hist2[jx, jy])
                    pmf_str = str(xbins[jx]) + ' \t' + str(ybins[jy]) + ' \t' + str(hist2[jx, jy]) + ' \t' + str(
                        dV_avg[jx, jy]) + ' \t' + str(dV_std[jx, jy]) + ' \t' + str(dV_anharm[jx, jy])
                    for k in range(int(nf_j)):
                        pmf_str = pmf_str + ' \t' + str(dV_mat[jx, jy, k])
                    pmf_str = pmf_str + '\n'
                    pmf_file.write(pmf_str)

    # PLOTTING FUNCTION FOR FREE ENERGY FIGURE
    if plt_figs:
        cbar_ticks = [0, Emax * .25, Emax * .5, Emax * .75, 8.0]
        plt.figure(2, figsize=(11, 8.5))
        extent = [newedgesX[0], newedgesX[-1], newedgesY[-1], newedgesY[0]]
        # print(extent)
        plt.imshow(hist2.transpose(), extent=extent, interpolation='gaussian')
        cb = plt.colorbar(ticks=cbar_ticks, format='% .1f', aspect=10)  # grab the Colorbar instance
        imaxes = plt.gca()
        plt.axes(cb.ax)
        plt.clim(vmin=0, vmax=8.0)
        plt.yticks(fontsize=18)
        plt.axes(imaxes)
        axis = (min(xbins), max(xbins), min(ybins), max(ybins))
        plt.axis(axis)
        plt.xticks(size='18')
        plt.yticks(size='18')
        plt.xlabel('RC1', fontsize=18)
        plt.ylabel('RC2', fontsize=18)
        plt.savefig('2D_Free_energy_surface.png', bbox_inches=0)
        # print("FIGURE SAVED 2D_Free_energy_surface.png")

        # PLOTTING FUNCTION FOR WEIGHTS histogram
        [hist, edges] = np.histogram(weights, bins=100)
        width = np.absolute(np.subtract(edges[0], edges[1]))
        plt.figure(1, figsize=(11, 8.5))
        plt.bar(edges[:100], hist, width=width, log=True)
        plt.yscale('log')  # if typerror is thrown delete .matplotlib/fontList.cache  file
        plt.xticks(fontsize='18')
        plt.yticks(fontsize='18')
        plt.savefig('weights.png', bbox_inches=0)
        # print("FIGURE SAVED weights.png")
