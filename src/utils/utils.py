import os
import shutil
import numpy as np


def find_euclidean_distance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance


def find_cosine_similarity(considered_representation, other_representations):
    return np.sum(other_representations * considered_representation, axis=1)


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def sort_images(labeled_names, save_path):
    for image_path, label in labeled_names:
        label_dir = os.path.join(save_path, f"person_{label}")

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        sep_pos = image_path.rfind("/") + 1
        image_name = image_path[sep_pos:]
        dest_path = os.path.join(label_dir, image_name)

        shutil.copyfile(image_path, dest_path)


def extract_person_name(labeled_image):
    image_path, label = labeled_image

    sep_pos = image_path.rfind("/") + 1
    dot_pos = image_path.rfind('.')
    image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
    person_name = image_name.split("_")[0]

    return image_path, person_name, label


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
