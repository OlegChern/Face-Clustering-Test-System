import numpy as np
from itertools import product
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.preprocessing import normalize
from src.utils.utils import evaluate_metrics


def cluster_kmeans(vectors, params_dict=None):
    vectors = np.asarray(vectors)
    if params_dict is None:
        params_dict = {"clusters": 4, "random_state": 170}

    evaluator = KMeans(n_clusters=params_dict["clusters"], random_state=params_dict["random_state"])

    return evaluator.fit_predict(vectors)


def cluster_dbscan(vectors, params_dict=None):
    vectors = np.asarray(vectors)
    if params_dict is None:
        params_dict = {"eps": 9, "min_samples": 3}

    evaluator = DBSCAN(eps=params_dict["eps"], min_samples=params_dict["min_samples"])

    return evaluator.fit_predict(vectors)


def cluster_mean_shift(vectors, params_dict=None):
    vectors = np.asarray(vectors)
    if params_dict is None:
        params_dict = {"bandwidth": 11}

    evaluator = MeanShift(bandwidth=params_dict["bandwidth"])

    return evaluator.fit_predict(vectors)


def cluster_threshold(vectors, params_dict):
    vectors = np.asarray(vectors)

    if params_dict is None:
        params_dict = {"threshold": 11.81}

    labels = []
    clusters = {}
    threshold = params_dict["threshold"]
    latest_cluster = 0

    for vector in vectors:
        added = False
        for cluster in clusters.keys():
            for labeled in clusters[cluster]:
                if np.linalg.norm(vector - labeled) <= threshold:
                    labels.append(cluster)
                    clusters[cluster].append(vector)
                    added = True
                    break

            if added:
                break
        else:
            clusters.update({latest_cluster: [vector]})
            labels.append(latest_cluster)
            latest_cluster += 1

    return labels


def optimal_params_grid_search(clusterer, algorithm, params_range):
    best_params_prec = dict()
    best_prec = 0

    best_params_rec = dict()
    best_rec = 0

    best_params_f1 = dict()
    best_f1 = 0

    for params in product(*params_range.values()):
        cur_dict = dict(zip(params_range.keys(), params))

        results = clusterer.cluster_images(algorithm, cur_dict)
        prec, rec, f1 = evaluate_metrics(results)

        if prec > best_prec:
            best_prec = prec
            best_params_prec = cur_dict
        if rec > best_rec:
            best_rec = rec
            best_params_rec = cur_dict
        if f1 > best_f1:
            best_f1 = f1
            best_params_f1 = cur_dict

    return best_prec, best_params_prec, best_rec, best_params_rec, best_f1, best_params_f1
