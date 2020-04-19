from src.clustering.clustering import ImageClusteringUnit
from src.clustering.algorithms.utils import sort_images, evaluate_metrics, optimal_params_grid_search
from src.test_system.logging import get_default_logger
from src.embedding.embeddings_creation import ImageEmbeddingsCreator
from timeit import default_timer

import os
from enum import Enum, auto


class Metric(Enum):
    PRECISION, RECALL, F1 = auto(), auto(), auto()


def evaluate_embeddings_creator(models, path_to_faces, logger=None):
    if logger is None:
        logger = get_default_logger("Embedding")

    for model, path in models.items():
        embeddings_creator = ImageEmbeddingsCreator(path_to_faces)

        start_time = default_timer()
        embeddings_creator.create_embeddings(model, path)
        end_time = default_timer()

        logger.info(f"Embeddings creation time for model {model.Name} is {end_time - start_time}s")


def evaluate_clustering_algorithms(algorithms_params_dict, embedding_path, results_path, metric=Metric.F1, logger=None):
    if logger is None:
        logger = get_default_logger("Clustering")

    clustering_unit = ImageClusteringUnit(embedding_path)

    params = dict()
    params.update({Metric.F1: None})
    params.update({Metric.PRECISION: None})
    params.update({Metric.RECALL: None})

    for name, (algorithm, params_range) in algorithms_params_dict.items():
        params[Metric.PRECISION], params[Metric.RECALL], params[Metric.F1] = \
            optimal_params_grid_search(clustering_unit, algorithm, params_range)

        start_time = default_timer()
        results = clustering_unit.cluster_images(algorithm, params_dict=params[metric])
        end_time = default_timer()

        path = results_path + f"/{name}"
        if not os.path.exists(path):
            os.mkdir(path)

        sort_images(results, path)
        prec, rec, f1 = evaluate_metrics(results)

        logger.info(
            f"Elapsed time for algorithm {name}: {end_time - start_time}s\n"
            f"Results: precision {prec}, recall {rec}, f1 {f1}\n"
            f"Maximised metric: {metric}\n"
            f"Chosen parameters: {params[metric]}")
