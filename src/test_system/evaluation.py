from src.utils.image_loader import ImageLoader
from src.clustering.clustering import ImageClusterer
from src.clustering.algorithms import optimal_params_grid_search
from src.utils.utils import sort_images, evaluate_metrics
from src.test_system.logging import get_default_logger
from src.embedding.embeddings_creation import facenet_create_embeddings
from timeit import default_timer

import os

results_dir = "./results/clustered"
embeddings_dir = "./results/embeddings/embeddings.txt"


def evaluate_facenet(model, path_to_faces, path_to_embeddings=embeddings_dir, logger=None):
    if logger is None:
        logger = get_default_logger()

    loader = ImageLoader(path_to_faces)

    start_time = default_timer()
    facenet_create_embeddings(loader, path_to_embeddings, model)
    end_time = default_timer()

    logger.info(f"Embedding creation time is {end_time - start_time}s")


def evaluate_clustering_algorithms(algorithms_params_dict, embedding_path=embeddings_dir, results_path=results_dir,
                                   metric="f1", logger=None):
    if logger is None:
        logger = get_default_logger()

    clusterer = ImageClusterer(embedding_path)

    for name, (algorithm, params_range) in algorithms_params_dict.items():
        _, best_params_prec, _, best_params_rec, _, best_params_f1 = optimal_params_grid_search(
            clusterer, algorithm, params_range)

        params = dict()
        params.update({"f1": best_params_f1})
        params.update({"recall": best_params_rec})
        params.update({"precision": best_params_prec})

        start_time = default_timer()
        results = clusterer.cluster_images(algorithm, params_dict=params[metric])
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


def evaluate_clustering_algorithm_with_optimal_params(algorithm, params_range, embedding_path=embeddings_dir,
                                                      logger=None):
    if logger is None:
        logger = get_default_logger()

    clusterer = ImageClusterer(embedding_path)

    best_prec, best_params_prec, best_rec, best_params_rec, best_f1, best_params_f1 = optimal_params_grid_search(
        clusterer, algorithm, params_range)

    logger.info(
        f"For this set:\n"
        f"Best possible precision is {best_prec} with params {best_params_prec}\n"
        f"Best possible recall is {best_rec} with params {best_params_rec}\n"
        f"best possible f1 score is {best_f1} with params {best_params_f1}\n")
