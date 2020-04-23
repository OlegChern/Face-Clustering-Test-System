# This is an implementation of https://arxiv.org/pdf/1604.00989.pdf, a modified version of rank-order clustering.

import pyflann
import numpy as np
from time import time
from functools import partial


def build_index(dataset, n_neighbors):
    """
    Takes a dataset, returns the "n" nearest neighbors
    """
    # Initialize FLANN
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset, algorithm='kdtree', trees=4)
    # print params
    nearest_neighbors, dists = flann.nn_index(dataset, n_neighbors, hecks=params['checks'])

    return nearest_neighbors, dists


def create_neighbor_lookup(nearest_neighbors):
    """
    Key is the reference face, values are the neighbors.
    """
    nn_lookup = {}
    for i in range(nearest_neighbors.shape[0]):
        nn_lookup[i] = nearest_neighbors[i, :]
    # print "NN Lookup :",nn_lookup
    return nn_lookup


def calculate_symmetric_dist_row(nearest_neighbors, nn_lookup, row_no):
    """
    This function calculates the symmetric distances for one row in the
    matrix.
    """
    dist_row = np.zeros([1, nearest_neighbors.shape[1]])
    f1 = nn_lookup[row_no]

    for idx, neighbor in enumerate(f1[1:]):
        Oi = idx + 1
        co_neighbor = True
        try:
            row = nn_lookup[neighbor]
            Oj = np.where(row == row_no)[0][0] + 1
        except IndexError:
            Oj = nearest_neighbors.shape[1] + 1
            co_neighbor = False

        # dij
        f11 = set(f1[0:Oi])
        f21 = set(nn_lookup[neighbor])
        dij = len(f11.difference(f21))
        # dji
        f12 = set(f1)
        f22 = set(nn_lookup[neighbor][0:Oj])
        dji = len(f22.difference(f12))

        # print 'dij: {}, dji: {}'.format(dij, dji)
        # print 'Oi: {}, Oj: {}'.format(Oi, Oj)

        if not co_neighbor:
            dist_row[0, Oi] = 9999.0
        else:
            dist_row[0, Oi] = float(dij + dji) / min(Oi, Oj)

    # print dist_row
    return dist_row


def calculate_symmetric_dist(app_nearest_neighbors):
    """
    This function calculates the symmetric distance matrix.
    """
    nn_lookup = create_neighbor_lookup(app_nearest_neighbors)
    d = np.zeros(app_nearest_neighbors.shape)

    func = partial(calculate_symmetric_dist_row, app_nearest_neighbors, nn_lookup)
    results = map(func, range(app_nearest_neighbors.shape[0]))

    for row_no, row_val in enumerate(results):
        d[row_no, :] = row_val

    return d


def aro_clustering(app_nearest_neighbors, distance_matrix, thresh):
    '''
    Approximate rank-order clustering. Takes in the nearest neighbors matrix
    and outputs clusters - list of lists.
    '''
    # Clustering :
    clusters = []
    # Start with the first face :
    nodes = set(list(np.arange(0, distance_matrix.shape[0])))
    # print 'Nodes initial : {}'.format(nodes)
    tc = time()
    plausible_neighbors = create_plausible_neighbor_lookup(app_nearest_neighbors, distance_matrix, thresh)
    # print 'Time to create plausible_neighbors lookup : {}'.format(time()-tc)
    ctime = time()
    while nodes:
        # Get a node :
        n = nodes.pop()

        # This contains the set of connected nodes :
        group = {n}

        # Build a queue with this node in it :
        queue = [n]

        # Iterate over the queue :
        while queue:
            n = queue.pop(0)
            neighbors = plausible_neighbors[n]
            # Remove neighbors we've already visited :
            neighbors = nodes.intersection(neighbors)
            neighbors.difference_update(group)

            # Remove nodes from the global set :
            nodes.difference_update(neighbors)

            # Add the connected neighbors :
            group.update(neighbors)

            # Add the neighbors to the queue to visit them next :
            queue.extend(neighbors)
        # Add the group to the list of groups :
        clusters.append(group)

    # print 'Clustering Time : {}'.format(time()-ctime)
    return clusters


def create_plausible_neighbor_lookup(app_nearest_neighbors, distance_matrix, thresh):
    """
    Create a dictionary where the keys are the row numbers(face numbers) and
    the values are the plausible neighbors.
    """
    n_vectors = app_nearest_neighbors.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        plausible_neighbors[i] = set(list(app_nearest_neighbors[i, np.where(distance_matrix[i, :] <= thresh)][0]))

    return plausible_neighbors


def cluster_app_rank_order(descriptor_matrix, params_dict=None):
    """
    Master function. Takes the descriptor matrix and returns clusters.
    n_neighbors are the number of nearest neighbors considered and thresh
    is the clustering distance threshold
    """
    if params_dict is None:
        params_dict = {"n_neighbors": 10, "threshold": 2}

    n_neighbors = params_dict["n_neighbors"]
    threshold = params_dict["threshold"]

    app_nearest_neighbors, dists = build_index(descriptor_matrix, n_neighbors)
    distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
    clusters_th = aro_clustering(app_nearest_neighbors, distance_matrix, threshold)

    labels = [0 for _ in range(len(descriptor_matrix))]
    for idx, cluster in enumerate(clusters_th):
        for image_idx in cluster:
            labels[image_idx] = idx

    return labels
