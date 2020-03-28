import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift


# Abstract Class
class ClusteringAlgorithm:
    Evaluator = None

    def fit_predict(self, vectors):
        vectors = np.asarray(vectors)

        return self.Evaluator.fit_predict(vectors)


class KmeansAlgorithm(ClusteringAlgorithm):

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"clusters": 5, "random_state": 170}

        self.Evaluator = KMeans(n_clusters=params_dict["clusters"], random_state=params_dict["random_state"])


class DbscanAlgorithm(ClusteringAlgorithm):

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"eps": 10, "min_samples": 2}

        self.Evaluator = DBSCAN(eps=params_dict["eps"], min_samples=params_dict["min_samples"])


class MeanShiftAlgorithm(ClusteringAlgorithm):

    def __init__(self, params_dict=None):
        if params_dict is None:
            params_dict = {"bandwidth": 10}

        self.Evaluator = MeanShift(bandwidth=params_dict["bandwidth"])
