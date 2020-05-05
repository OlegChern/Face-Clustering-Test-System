from src.clustering.clustering import ImageClusteringUnit
from src.clustering.clustering_utils import evaluate_metrics, optimal_params_grid_search
from src.image_processing.image_utils import sort_images
from src.image_processing.image_loader import ImageLoader
from src.test_system.logging import get_default_logger
from src.embedding.embeddings_creation import ImageEmbeddingsCreator
from src.extraction.mtcnn_face_extraction import FaceExtractorMTCNN
from timeit import default_timer

import os
from enum import Enum, auto


class Metric(Enum):
    PRECISION, RECALL, F1 = auto(), auto(), auto()


def evaluate_mtcnn_alignment(images_path, save_path, aligners, logger=None):
    if logger is None:
        logger = get_default_logger("Alignment")

    loader = ImageLoader(images_path)
    extractor = FaceExtractorMTCNN()
    for aligner in aligners:
        name = aligner.Name if aligner is not None else "None"

        target_path = save_path + "/" + name
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        extractor.extract_faces(loader, target_path, aligner)

        logger.info(f"{name} alignment for the image set took {extractor.AlignmentTime}s")


def evaluate_embeddings_creator(models, path_to_faces, logger=None):
    if logger is None:
        logger = get_default_logger("Embedding")

    for model_constructor, weight_path, embeddings_path in models:
        embeddings_creator = ImageEmbeddingsCreator(path_to_faces)

        start_time = default_timer()
        model = model_constructor(weight_path)
        end_time = default_timer()

        init_time = end_time - start_time

        start_time = default_timer()
        embeddings_creator.create_embeddings(model, embeddings_path)
        end_time = default_timer()

        embed_time = end_time - start_time

        logger.info(f"Model {model.Name} initialized in {init_time}s and embeddings created in {embed_time}s")


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
