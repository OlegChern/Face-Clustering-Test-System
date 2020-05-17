from src.clustering.clustering_utils import find_euclidean_distance, find_cosine_similarity
from dlib import chinese_whispers_clustering
import dlib


def cluster_threshold(vectors, params_dict):
    if params_dict is None:
        params_dict = {"threshold": 0.15, "distance": find_euclidean_distance}

    labels = []
    clusters = {}
    threshold = params_dict["threshold"]
    distance_func = params_dict["distance"]
    latest_cluster = 0

    for vector in vectors:
        added = False
        for cluster in clusters.keys():
            for labeled in clusters[cluster]:
                if distance_func(vector, labeled) <= threshold:
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


def chinese_whispers(encodings, params_dict):
    encodings = [dlib.vector(vector) for vector in encodings]
    threshold = params_dict["threshold"]
    labels = chinese_whispers_clustering(encodings, threshold)

    return labels
