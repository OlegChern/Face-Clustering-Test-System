from sklearn.cluster import KMeans, DBSCAN, MeanShift, AffinityPropagation, SpectralClustering

from src.clustering.utils import find_euclidean_distance


def cluster_kmeans(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"clusters": 4, "random_state": 170}

    evaluator = KMeans(**params_dict)

    return evaluator.fit_predict(vectors)


def cluster_dbscan(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"eps": 1, "min_samples": 1, "metric": find_euclidean_distance}

    evaluator = DBSCAN(**params_dict)

    return evaluator.fit_predict(vectors)


def cluster_mean_shift(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"bandwidth": 6.3}

    evaluator = MeanShift(**params_dict)

    return evaluator.fit_predict(vectors)


def cluster_affinity_propagation(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"damping": 0.5}

    evaluator = AffinityPropagation(**params_dict)

    return evaluator.fit_predict(vectors)


def cluster_spectral(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"clusters": 4, "random_state": 170}

    evaluator = SpectralClustering(**params_dict)

    return evaluator.fit_predict(vectors)
