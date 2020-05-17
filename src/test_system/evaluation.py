from src.clustering.clustering import ImageClusteringUnit
from src.test_system.evaluation_utils import evaluate_pairwise_metrics, optimal_params_grid_search
from src.image_processing.image_utils import sort_images
from src.image_processing.image_loader import ImageLoader
from src.test_system.logging import get_default_logger
from src.embedding.embeddings_creation import ImageEmbeddingsCreator
from timeit import default_timer

import os
from enum import Enum, auto


class Metric(Enum):
    PRECISION, RECALL, F1 = auto(), auto(), auto()


def evaluate_normalizers(images_path, save_path, extractors, normalizers, logger=None):
    if logger is None:
        logger = get_default_logger("Alignment")

    for extractor in extractors:
        for normalizer in normalizers:
            loader = ImageLoader(images_path)

            name = normalizer.Name if normalizer is not None else "No"
            extractor.extract_faces(loader, save_path, normalizer)

            logger.info(
                f"{name} normalization of {extractor.ExtractedFaces} faces "
                f"for {extractor.Name} took {extractor.NormalizationTime}s")


def evaluate_embeddings_creator(models, faces_path, embeddings_dir, logger=None):
    if logger is None:
        logger = get_default_logger("Embedding")

    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    for model_constructor, kwargs in models:
        embeddings_creator = ImageEmbeddingsCreator(faces_path)

        start_time = default_timer()
        model = model_constructor(**kwargs)
        end_time = default_timer()

        init_time = end_time - start_time

        result_path = f"{embeddings_dir}/{model.Name}.txt"
        embeddings_creator.create_embeddings(model, result_path)

        logger.info(
            f"Model {model.Name} initialized in {init_time}s, "
            f"image prepocessing took {embeddings_creator.ImagePreprocessingTime}s "
            f"and embeddings creation took {embeddings_creator.EmbeddingsCreationTime}s")


def evaluate_clustering_algorithms(algorithms_params_dict, embedding_path, results_path=None, top_n=None, logger=None,
                                   n_threads=1, inter_logging=False):
    if logger is None:
        logger = get_default_logger("Clustering")

    clustering_unit = ImageClusteringUnit(embedding_path)
    vectors_num = clustering_unit.get_total_vectors_number() if top_n is None else top_n

    for name, (algorithm, params_range) in algorithms_params_dict.items():
        results_dict = optimal_params_grid_search(clustering_unit, algorithm, name, params_range, top_n, n_threads,
                                                  inter_logging)

        results = results_dict["results"]

        if results is None:
            continue

        if results_path is not None:
            path = results_path + f"/{name}"
            if not os.path.exists(path):
                os.mkdir(path)

            sort_images(results, path)

        prec = results_dict["precision"]
        rec = results_dict["recall"]
        f1 = results_dict["f1-measure"]
        fp = results_dict["false positives"]
        time = results_dict["time"]
        params = results_dict["params"]

        logger.info(
            f"Elapsed time for {vectors_num} vectors and algorithm {name}: {time}s\n"
            f"Results: precision {prec}, recall {rec}, f1 {f1}, false positives {fp}\n"
            f"Chosen parameters: {params}")
