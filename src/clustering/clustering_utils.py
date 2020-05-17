import numpy as np
from numpy.linalg import norm


def find_euclidean_distance(considered_representation, other_representations):
    euclidean_distance = other_representations - considered_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance),
                                axis=other_representations.ndim - 1)
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance


def find_cosine_similarity(considered_representation, other_representations):
    cosine_similarity = np.sum(np.multiply(other_representations, considered_representation),
                               axis=other_representations.ndim - 1)
    cosine_similarity = cosine_similarity / (norm(other_representations) * norm(considered_representation))
    cosine_similarity = 1 - (cosine_similarity + 1) / 2

    return cosine_similarity


def find_taxicab_distance(considered_representation, other_representations):
    taxicab_distance = other_representations - considered_representation
    taxicab_distance = np.abs(taxicab_distance)
    taxicab_distance = np.sum(taxicab_distance, axis=other_representations.ndim - 1)

    return taxicab_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x), axis=x.ndim - 1))[:, None]


