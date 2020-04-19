import numpy as np
import networkx as nx

from random import shuffle
from src.clustering.algorithms.utils import find_euclidean_distance, find_cosine_similarity


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


# https://github.com/zhly0/facenet-face-cluster-chinese-whispers-/
def chinese_whisperers(encodings, params_dict=None):
    if params_dict is None:
        params_dict = {"threshold": 0.18, "iterations": 5, "distance": find_cosine_similarity}

    threshold = params_dict["threshold"]
    iterations = params_dict["iterations"]
    distance_func = params_dict["distance"]

    nodes, edges = [], []
    indices = range(0, len(encodings))
    for idx, face_encoding_to_check in enumerate(encodings):
        node_id = idx + 1

        node = (node_id, {'cluster': indices[idx], 'vector_num': indices[idx]})
        nodes.append(node)

        if node_id == len(encodings):
            break

        compare_encodings = encodings[node_id:]
        distances = distance_func(face_encoding_to_check, compare_encodings)
        for i, distance in enumerate(distances):
            if distance < threshold:
                edge_id = node_id + i + 1
                edges.append((node_id, edge_id, {'weight': distance}))

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    for _ in range(iterations):
        cluster_nodes = graph.nodes()
        shuffle(np.array(cluster_nodes))
        for node in cluster_nodes:
            neighbors = graph[node]
            clusters = {}

            for ne in neighbors:
                if graph.nodes[ne]['cluster'] in clusters:
                    clusters[graph.nodes[ne]['cluster']] += graph[node][ne]['weight']
                else:
                    clusters[graph.nodes[ne]['cluster']] = graph[node][ne]['weight']

            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            graph.nodes[node]['cluster'] = max_cluster

    clusters = [0 for _ in range(len(encodings))]
    for (_, data) in graph.nodes.items():
        cluster = data['cluster']
        vector_num = data['vector_num']

        clusters[vector_num] = cluster

    return clusters
