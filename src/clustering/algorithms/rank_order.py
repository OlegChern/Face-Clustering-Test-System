import pyflann
import numpy as np

from itertools import combinations, product


def build_index(vectors, distance="euclidean"):
    pyflann.set_distance_type(distance_type=distance)

    flann = pyflann.FLANN()
    params = flann.build_index(vectors, algorithm='linear')

    order_list, dists = flann.nn_index(vectors, len(vectors), checks=params['checks'])
    order_list.astype(int)

    return order_list, dists


def create_neighbor_lookups(fs, dists):
    os = np.zeros(fs.shape, dtype=int)
    face_dists = np.zeros(fs.shape)

    for i, j in product(range(len(fs)), repeat=2):
        face_dists[i][fs[i][j]] = dists[i][j]
        os[i][fs[i][j]] = j

    return os, face_dists


def cluster_rank_order(vectors, params_dict=None):
    distance = params_dict["distance"]
    k = params_dict["k_neighbors"]
    threshold = params_dict["threshold"]

    fs, dists = build_index(vectors, distance=distance)
    os, faces_dists = create_neighbor_lookups(fs, dists)
    clusters_dists = faces_dists.copy()

    clusters = dict({i: [i] for i in range(len(vectors))})
    merged_into = [i for i in range(len(clusters))]

    def rank_order_distance(c1, c2):
        d_c1_c2 = np.sum(os[c2][fs[c1][i]] for i in range(os[c1][c2]))
        d_c2_c1 = np.sum(os[c1][fs[c2][i]] for i in range(os[c2][c1]))

        return (d_c1_c2 + d_c2_c1) / min(os[c1][c2], os[c2][c1])

    def rank_order_distance_normalized(c1, c2):
        return (1 / phi(c1, c2)) * clusters_dists[c1][c2]

    def phi(c1, c2):
        result = 1 / (len(clusters[c1]) + len(clusters[c2]))
        result *= np.sum(
            (1 / k) * np.sum(faces_dists[a][fs[a][i]] for i in range(1, k + 1)) for a in clusters[c1] + clusters[c2])

        return result

    to_be_merged = True
    while to_be_merged:
        to_be_merged = False
        merging_candidates = []
        for ci, cj in combinations(clusters.keys(), 2):
            if rank_order_distance(ci, cj) < threshold and rank_order_distance_normalized(ci, cj) < 1:
                to_be_merged = True
                merging_candidates.append((ci, cj))

        for ci, cj in merging_candidates:
            while merged_into[ci] != ci or merged_into[cj] != cj:
                ci = merged_into[ci]
                cj = merged_into[cj]

            if ci == cj:
                continue

            clusters[ci] += clusters[cj]
            clusters.pop(cj)
            merged_into[cj] = ci

            lower_indices = np.where(clusters_dists[cj] < clusters_dists[ci])
            clusters_dists[ci][lower_indices] = clusters_dists[cj][lower_indices]

            fs[ci] = np.argsort(clusters_dists[ci])

            for i in range(len(fs)):
                os[ci][fs[ci][i]] = i

    labels = [0 for _ in range(len(vectors))]
    for idx, cluster in clusters.items():
        for image_idx in cluster:
            labels[image_idx] = idx

    return labels
