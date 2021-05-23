import numpy as np
import matplotlib.pyplot as plt


def gromos(rmsd, threshold, nmax):
    def sort_clusters(cluster_dict):
        cluster_list = [(the_medoid, members) for the_medoid, members in cluster_dict.items()]
        return sorted(cluster_list, key=lambda x: len(x[1]), reverse=True)

    def gromos_inner(the_rmsd, the_threshold, n_max):
        # gromos clustering Daura et al. (Angew. Chem. Int. Ed. 1999, 38, pp 236-240)
        threshold_matrix = the_rmsd < the_threshold
        number_of_elements = np.count_nonzero(threshold_matrix, axis=0)
        the_medoid_index = np.argmax(number_of_elements)
        members = threshold_matrix[the_medoid_index, :]
        not_members = np.invert(members)
        the_member_indices = members.nonzero()[0]
        not_members_indices = not_members.nonzero()[0]
        result = [(the_medoid_index, the_member_indices)]
        if n_max > 1 and len(not_members_indices) > 0:
            other_results = gromos_inner(the_rmsd[not_members, :][:, not_members], the_threshold, n_max - 1)
            for o in other_results:
                result.append((not_members_indices[o[0]], not_members_indices[o[1]]))
        return result
    dimension = rmsd.shape[0]
    unassigned = set(range(dimension))
    the_result = gromos_inner(rmsd, threshold, nmax)
    assigned = {}
    for medoid_index, member_indices in the_result:
        assigned[medoid_index] = list(member_indices)
        unassigned.difference_update(set(member_indices))
    assigned = sort_clusters(assigned)
    xx = list(unassigned)
    yy = np.full((len(xx),), 0, dtype=float)
    fig, ax1 = plt.subplots()
    ax1.scatter(xx, yy, alpha=0.2, color='red')
    if xx:
        labels = ['None']
        labels2 = [str(len(xx))]
    else:
        labels = ['']
        labels2 = ['']
    ticks = [0.0]
    for i, (medoid, cluster) in enumerate(assigned):
        if i > 16:
            continue
        labels.append(str(medoid))
        labels2.append(str(len(cluster)))
        ticks.append(float(i+1))
        x = [frame for frame in cluster]  # nanoseconds
        y = np.full((len(x),), i+1, dtype=float)
        ax1.scatter(x, y, alpha=0.2, color='black')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Cluster medoid")
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(labels)
    if xx:
        ax1.set_ylim(-1, len(labels))
    else:
        ax1.set_ylim(0, len(labels))
    ax2 = ax1.twinx()
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels2)
    if xx:
        ax2.set_ylim(-1, len(labels))
    else:
        ax2.set_ylim(0, len(labels))
    ax2.set_ylabel("Number of frames")
    plt.show()


if __name__ == '__main__':
    matrix = np.loadtxt('/home/groups/eberini/Merck/MD_giulia/analysis2/adam_g0f_rmsdat.dat')

    gromos(matrix, 25, 15)


