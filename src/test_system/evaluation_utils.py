from functools import reduce, partial
from multiprocessing import Pool
from itertools import combinations, product
from timeit import default_timer

from tqdm import tqdm

from src.image_processing.image_utils import extract_person_name
from src.test_system.logging import get_file_logger


def evaluate_pairwise_metrics(labeled_images):
    labeled_images = list(map(extract_person_name, labeled_images))

    true_positive = 0
    false_negative = 0
    false_positive = 0
    total_predictions = 0

    for image1, image2 in combinations(labeled_images, r=2):
        total_predictions += 1

        path1, person1, label1 = image1
        path2, person2, label2 = image2

        if person1 == person2:
            if label1 == label2:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if label1 == label2:
                false_positive += 1

    if true_positive == 0:
        return 0, 0, 0, false_positive / total_predictions

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, false_positive / total_predictions


def optimal_params_grid_search(clusterer, algorithm, algorithm_name, params_range, top_n=None, n_threads=1,
                               inter_logging=False):
    params_grid = product(*params_range.values())
    combinations_total = reduce(lambda acc, x: acc * len(x), list(params_range.values()), 1)

    mapping_evaluate_and_log = partial(evaluate_and_log, clusterer, list(params_range.keys()), algorithm, top_n,
                                       inter_logging, algorithm_name)

    with Pool(processes=n_threads) as p:
        result_dicts = list(tqdm(p.imap(mapping_evaluate_and_log, params_grid), total=combinations_total))

    best = max(result_dicts, key=lambda x: x["f1-measure"])

    return best


def evaluate_and_log(clusterer, params_names, algorithm, top_n, inter_logging, algorithm_name, params):
    cur_dict = dict(zip(params_names, params))

    inter_logger = None
    if inter_logging:
        inter_logger = get_file_logger(log_name=f"{algorithm_name}_temp")

    start = default_timer()
    results = clusterer.cluster_images(algorithm, cur_dict, top_n=top_n)
    end = default_timer()

    prec, rec, f1, fp = evaluate_pairwise_metrics(results)
    evaluation_time = end - start

    if inter_logger is not None:
        inter_logger.info(f"precision {prec} recall {rec} f1 {f1} false positives {fp}\nparams {params}")

    result_dict = {"false positives": fp, "precision": prec, "f1-measure": f1, "recall": rec, "params": params,
                   "time": evaluation_time, "results": results}

    return result_dict
