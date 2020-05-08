from src.clustering.clustering import ImageClusteringUnit
from src.clustering.clustering_utils import evaluate_metrics, optimal_params_grid_search
from src.image_processing.image_utils import sort_images
from src.image_processing.image_loader import ImageLoader
from src.test_system.logging import get_default_logger
from src.embedding.embeddings_creation import ImageEmbeddingsCreator
from src.extraction.face_extraction import FaceExtractorMTCNN
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

            logger.info(f"{name} normalization for {extractor.Name} took {extractor.NormalizationTime}s")


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


def evaluate_clustering_algorithms(algorithms_params_dict, embedding_path, results_path, metric=Metric.F1, logger=None):
    if logger is None:
        logger = get_default_logger("Clustering")

    clustering_unit = ImageClusteringUnit(embedding_path)

    params = dict()
    params.update({Metric.F1: None})
    params.update({Metric.RECALL: None})
    params.update({Metric.PRECISION: None})

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
