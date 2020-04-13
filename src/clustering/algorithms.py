import numpy as np
import networkx as nx

from random import shuffle
from itertools import product
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from src.utils.utils import evaluate_metrics, find_euclidean_distance, find_cosine_similarity


def cluster_kmeans(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"clusters": 4, "random_state": 170}

    evaluator = KMeans(n_clusters=params_dict["clusters"], random_state=params_dict["random_state"])

    return evaluator.fit_predict(vectors)


def cluster_dbscan(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"eps": 1, "min_samples": 1}

    evaluator = DBSCAN(eps=params_dict["eps"], min_samples=params_dict["min_samples"])

    return evaluator.fit_predict(vectors)


def cluster_mean_shift(vectors, params_dict=None):
    if params_dict is None:
        params_dict = {"bandwidth": 6.3}

    evaluator = MeanShift(bandwidth=params_dict["bandwidth"])

    return evaluator.fit_predict(vectors)


def cluster_threshold(vectors, params_dict):
    if params_dict is None:
        params_dict = {"threshold": 0.15}

    labels = []
    clusters = {}
    threshold = params_dict["threshold"]
    latest_cluster = 0

    for vector in vectors:
        added = False
        for cluster in clusters.keys():
            for labeled in clusters[cluster]:
                if find_euclidean_distance(vector, labeled) <= threshold:
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


# https://github.com/zhly0/facenet-face-cluster-chinese-whispers-/
def chinese_whisperers(encodings, params_dict=None):
    if params_dict is None:
        params_dict = {"threshold": 0.18, "iterations": 5}

    threshold = params_dict["threshold"]
    iterations = params_dict["iterations"]

    # Create graph
    nodes = []
    edges = []

    indices = range(0, len(encodings))
    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx + 1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': indices[idx], 'path': indices[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx + 1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx + 1:]
        distances = find_cosine_similarity(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = graph.nodes()
        shuffle(np.array(cluster_nodes))
        for node in cluster_nodes:
            neighbors = graph[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if graph.nodes[ne]['cluster'] in clusters:
                        clusters[graph.nodes[ne]['cluster']] += graph[node][ne]['weight']
                    else:
                        clusters[graph.nodes[ne]['cluster']] = graph[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            graph.nodes[node]['cluster'] = max_cluster

    clusters = [0 for _ in range(len(encodings))]

    # Prepare cluster output
    for (_, data) in graph.nodes.items():
        cluster = data['cluster']
        path = data['path']

        clusters[path] = cluster

    return clusters


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
