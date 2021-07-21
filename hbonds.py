import pickle
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import statsmodels.api as sm
lowess = sm.nonparametric.lowess


plt.style.use('seaborn-darkgrid')

# here set the number of different plots
colors = sns.color_palette("hls", 9)


def process_hbonds(raw_bonds, topology, sel1='all', sel2='all', freq=5, join_residue=True):
    # group residues

    the_top = md.load(topology).topology
    selection1 = the_top.select(sel1)
    selection2 = the_top.select(sel2)

    def process_label(the_label):
        splitted = the_label.split(' -- ')
        res1 = splitted[0].split('-')[0]
        res2 = splitted[1].split('-')[0]
        return f'{res1} -- {res2}'
    times = []
    hbonds = {}

    for frame in raw_bonds:
        the_time, the_hbonds = frame
        times.append(the_time)
        for bond in the_hbonds:
            if ((bond[0] in selection1) and (bond[2] in selection2) or
                    (bond[2] in selection1) and (bond[0] in selection2)):
                label = f'{the_top.atom(bond[0])} -- {the_top.atom(bond[2])}'
                if label in hbonds.keys():
                    hbonds[label].append(the_time)
                else:
                    hbonds[label] = [the_time]

    # filter frequency
    threshold = len(times) / 100 * freq
    hb_filtered = {lab: hbonds[lab] for lab in hbonds.keys() if len(hbonds[lab]) >= threshold}
    hb_labels = sorted(list(hb_filtered.keys()))
    hb_grouped_labels = sorted(list({process_label(label) for label in hb_labels}))
    hb_grouped_labels_dict = {label: process_label(label) for label in hb_labels}

    hb_matrix = np.zeros((len(hb_labels), len(times)))
    hb_grouped_matrix = np.zeros((len(hb_grouped_labels), len(times)))
    for label in hb_labels:
        for the_time in hbonds[label]:
            hb_matrix[hb_labels.index(label), times.index(the_time)] = 1
            hb_grouped_matrix[hb_grouped_labels.index(hb_grouped_labels_dict[label]), times.index(the_time)] += 1

    if join_residue:
        return hb_grouped_labels, times, hb_grouped_matrix
    else:
        return hb_labels, times, hb_matrix


def plot_num_bonds(the_times, the_matrix, title=None, width=10):
    num_bonds = the_matrix.sum(axis=0)
    the_times = np.array(the_times)
    if np.max(the_times) > 10000:
        xlabel = 'time (ns)'
        the_times /= 1000.0
    else:
        xlabel = 'time (ps)'
    if title:
        plt.title(title)
    plt.plot(the_times, num_bonds)

    plt.xlabel(xlabel)
    plt.ylabel('# hbonds')
    plt.tight_layout()
    plt.show()


def plot_many_bonds(the_times, the_matrices, the_labels, the_colors, title=None, alpha=0.1, width=10, divisor=1.0):
    fig = plt.figure(figsize=(width, width / 2))
    xlabel = 'time (ps)'
    times = []
    for tim in the_times:
        if np.max(tim) > 10000:
            xlabel = 'time (ns)'
    for tim in the_times:
        if xlabel == 'time (ps)':
            times.append(tim)
        else:
            times.append([t / 1000.0 for t in tim])

    num_bonds = []
    for mat in the_matrices:
        num_bonds.append(mat.sum(axis=0)/divisor)

    fig = plt.figure(figsize=(width, width/2), constrained_layout=True)
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle('color', colors)

    if title:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel('# hbonds')

    for t, nb, lab, color in zip(times, num_bonds, the_labels, the_colors):
        plt.scatter(t, nb, alpha=alpha, color=color)

    for t, nb, lab, color in zip(times, num_bonds, the_labels, the_colors):
        y = lowess(nb, t, return_sorted=True, frac=0.5)
        plt.plot(y[:, 0], y[:, 1], linewidth=4, label=lab, color=color)

    y = ax.get_yaxis()
    y.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.show()


def plot_bonds(the_labels, the_times, the_matrix, width=10, height=None, title=None):
    if not height:
        height = 0.2 * len(the_labels) + 1
    the_times = np.array(the_times)
    if np.max(the_times) > 10000:
        xlabel = 'time (ns)'
        the_times /= 1000.0
    else:
        xlabel = 'time (ps)'
    extent = [the_times[0], the_times[-1], -.5, the_matrix.shape[0] - .5]

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if np.max(the_matrix) > 1.0:
        cmap = plt.get_cmap('brg', np.max(the_matrix))
        cmap.set_under('w')
        plot = ax.imshow(the_matrix, aspect='auto', cmap=cmap, interpolation='nearest', extent=extent,
                         vmin=np.min(the_matrix)+0.5, vmax=np.max(the_matrix)+.5, origin='lower')
        cbar = fig.colorbar(plot, ticks=np.arange(np.min(the_matrix), np.max(the_matrix)+1), extend='min')
        cbar.set_label('# of hbonds')
    else:
        ax.imshow(the_matrix, aspect='auto', cmap='binary', interpolation='nearest', extent=extent, origin='lower')
        print(np.max(the_matrix), np.min(the_matrix))
    ax.set_yticks(list(range(len(the_labels))))
    ax.set_yticklabels(the_labels)
    ax.set_xlabel(xlabel)
    if title:
        plt.title(title)

    plt.tight_layout()

    plt.show()


def save_labels(the_labels, filename):
    with open(filename, 'wt') as out:
        for label in the_labels:
            out.write(label+'\n')


if __name__ == '__main__':
    tmp = '/home/groups/eberini/Merck/AMBER/analysis/g0f_rep1_hbonds.pickle'
    top = '/home/groups/eberini/Merck/AMBER/analysis/g0f_nowat.pdb'

    raw_hbonds = pickle.load(open(tmp, 'rb'))
    labels, time, matrix = process_hbonds(raw_hbonds, top, sel1='protein', sel2='resSeq 1331 to 1338',
                                          freq=0, join_residue=True)
    plot_num_bonds(time, matrix, 'Pippo\'s Hbonds')
    save_labels(labels, 'pippo.txt')
