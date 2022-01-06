import networkx as nx
import pymetis
from astropy.table import Table
from scipy.spatial import distance_matrix
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

tab = Table.read('test_images/scopesim_cluster.dat', format='ascii.ecsv')
def graph_test():

    xy = np.array((tab['x'],tab['y'])).T
    d_mat = distance_matrix(xy,xy)

    d_mat[d_mat > 50] = 0

    graph = nx.from_numpy_matrix(d_mat)
    #S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    #biggest = S[0]

    def flatten(l: list[list[T]]) -> list[T]:
        return [entry for sublist in l for entry in sublist]

    def to_csr(adj_list):
        xadj=[0]
        adjncy=[]
        for sublist in adj_list:
            xadj.append(xadj[-1]+len(sublist))
            adjncy+=sublist
        return xadj, adjncy


    #ncuts, membership = pymetis.part_graph(50, adj_list)
    adj_list = [list(i) for i in graph.adj.values()]
    weights = [[int(1000/i['weight']) for i in info.values()] for info in graph.adj.values()]
    xadj, adjncy = to_csr(adj_list)
    ncuts, membership = pymetis.part_graph(30, xadj=xadj, adjncy=adjncy, eweights=flatten(weights))

    print(nx.number_connected_components(graph))
    print(np.unique([len(i) for i in list(nx.connected_components(graph))]))

    layout = dict(zip(range(len(xy)), xy))
    nx.draw(graph, layout, node_color=membership, cmap='prism')

def recursive_kmeans(xy, max_size):

    avg_size = len(xy)/max_size
    initial = KMeans(n_clusters=int(avg_size/2)).fit(xy)

    labels = initial.labels_

    while True:
        print('loop')
        unique_labels, group_sizes = np.unique(labels, return_counts=True)
        greater = group_sizes > max_size
        if np.any(greater):
            for label, group_size in zip(unique_labels[greater], group_sizes[greater]):
                subdivide = KMeans(n_clusters=int(np.ceil(group_size/max_size))).fit(xy[labels == label])
                labels[labels == label] = subdivide.labels_ + np.max(labels) + 1
        else:
            break

    return labels


def cluster_test():
    tab = Table.read('test_images/scopesim_cluster.dat', format='ascii.ecsv')

    xy = np.array((tab['x'],tab['y'])).T
    n_clusters = int(len(xy)/5)

    #kmeans = AgglomerativeClustering(n_clusters=n_clusters).fit(xy)
    labels = recursive_kmeans(xy, 10)

    plt.scatter(tab['x'], tab['y'], c=labels, cmap='prism')
    _, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.hist(counts, bins=20)

if __name__ == '__main__':
    cluster_test()
    plt.show()