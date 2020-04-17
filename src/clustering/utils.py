import os
import shutil
from itertools import product

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


def evaluate_metrics(labeled_images):
    labeled_images = list(map(extract_person_name, labeled_images))

    true_positive = 0
    false_negative = 0
    false_positive = 0
    for current_path, current_person, current_label in labeled_images:
        for path, person, label in labeled_images:
            if path == current_path:
                continue

            if current_person == person:
                if current_label == label:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if current_label == label:
                    false_positive += 1

    if true_positive == 0 and (false_positive == 0 or false_negative == 0):
        return 0, 0, 0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


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


def extract_person_name(labeled_image):
    image_path, label = labeled_image

    sep_pos = image_path.rfind("/") + 1
    dot_pos = image_path.rfind('.')
    image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
    person_name = image_name.split("_")[0]

    return image_path, person_name, label


def sort_images(labeled_names, save_path):
    for image_path, label in labeled_names:
        label_dir = os.path.join(save_path, f"person_{label}")

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        sep_pos = image_path.rfind("/") + 1
        image_name = image_path[sep_pos:]
        dest_path = os.path.join(label_dir, image_name)

        shutil.copyfile(image_path, dest_path)
