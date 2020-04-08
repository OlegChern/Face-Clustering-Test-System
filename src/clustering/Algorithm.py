import numpy as np
from itertools import product
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from src.utils.Utils import evaluate_metrics


# Abstract Class
class ClusteringAlgorithm:
    Evaluator = None
    Name = "Default_Name"

    def fit_predict(self, vectors):
        vectors = np.asarray(vectors)

        return self.Evaluator.fit_predict(vectors)


class KmeansAlgorithm(ClusteringAlgorithm):
    Name = "K_Means"

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"clusters": 4, "random_state": 170}

        self.Evaluator = KMeans(n_clusters=params_dict["clusters"], random_state=params_dict["random_state"])


class DbscanAlgorithm(ClusteringAlgorithm):
    Name = "DBSCAN"

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"eps": 10, "min_samples": 2}

        self.Evaluator = DBSCAN(eps=params_dict["eps"], min_samples=params_dict["min_samples"])


class MeanShiftAlgorithm(ClusteringAlgorithm):
    Name = "Mean_Shift"

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"bandwidth": 10}

        self.Evaluator = MeanShift(bandwidth=params_dict["bandwidth"])


def optimal_params_grid_search(clusterer, algorithm, params_range):
    best_params_prec = dict()
    best_prec = 0

    best_params_rec = dict()
    best_rec = 0

    best_params_f1 = dict()
    best_f1 = 0

    for params in product(*params_range.values()):
        cur_dict = dict(zip(params_range.keys(), params))
        cur_algorithm = algorithm(cur_dict)

        results = clusterer.cluster_images(cur_algorithm)
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
