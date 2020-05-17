from src.clustering.algorithms.app_rank_order import cluster_app_rank_order
from src.clustering.algorithms.rank_order import cluster_rank_order
from src.clustering.algorithms.scikit_algorithms import cluster_mean_shift, cluster_dbscan, cluster_kmeans, \
    cluster_affinity_propagation, cluster_spectral, cluster_agglomerative, cluster_optics
from src.image_processing.image_loader import ImageLoader
from src.clustering.clustering_utils import find_taxicab_distance, find_cosine_similarity, find_euclidean_distance
from src.embedding.models_code.face_net import FaceNet
from src.embedding.models_code.open_face import OpenFace
from src.embedding.models_code.vgg_face import FaceVGG16, FaceVGGResNet, FaceVGGSqueezeNet
from src.test_system.evaluation import evaluate_normalizers, evaluate_embeddings_creator, \
    evaluate_clustering_algorithms
from src.extraction.face_normalization import EyesNoseAligner, EyesOnlyAligner, MappingAligner, FaceCropperVGG
from src.clustering.algorithms.algorithms import chinese_whispers, cluster_threshold, chinese_whispers_dlib
from src.extraction.face_extraction import FaceExtractorDlib, FaceExtractorMTCNN, FaceExtractorLFW
from src.test_system.logging import get_file_logger

from timeit import default_timer
from tqdm import trange

import numpy as np
import cv2

embeddings_dir = "./results/embeddings"
save_path = "./results/extraction/LFW"
# save_path = "./lfw/images/test"
load_path = "./results/extraction/LFW/Dlib-based Face Extractor/Dlib Mapping Aligner"
images_path = "./results/extraction/LFW/lfw-mtcnn-aligned"
# images_path = "./lfw/images/test/test"
sorted_path = "./results/clustered"

facenet_sandberg_weights = "./models/facenet_david_sandberg/facenet_weights.h5"
facenet_hiroki_weights = "./models/facenet_hiroki_taniai/facenet_hiroki_weights.h5"
openface_weights = "./models/open_face/openface_weights.h5"
vgg_face_weights = "./models/vgg_face/vgg_face_weights.h5"


def test_extraction_and_alignment():
    logger = get_file_logger()

    extractors = list()
    # extractors.append(FaceExtractorMTCNN())
    # extractors.append(FaceExtractorDlib())
    extractors.append(FaceExtractorLFW())

    normalizers = list()
    normalizers.append(EyesOnlyAligner(left_eye=(0.35, 0.35)))
    normalizers.append(EyesNoseAligner())
    normalizers.append(MappingAligner())
    normalizers.append(FaceCropperVGG())

    evaluate_normalizers(images_path, save_path, extractors, normalizers, logger=logger)


def test_embeddings_creation(faces_path, result_path):
    logger = get_file_logger()

    models = list()
    # models.append((FaceNet, {"weights_path": facenet_sandberg_weights}))
    # models.append((OpenFace, {"weights_path": openface_weights}))
    # models.append((FaceVGGSqueezeNet, {}))
    # models.append((FaceVGGResNet, {}))
    models.append((FaceVGG16, {}))

    evaluate_embeddings_creator(models, faces_path, result_path, logger=logger)


def elijah_test_1():
    embedding_file = embeddings_dir + "/LFW/Dlib Mapping Aligner/OpenFace.txt"
    algorithms = dict()

    threshold_range = {"threshold": [0.48], "distance": [find_euclidean_distance]}
    algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

    # chinese_whisperers_range = {"threshold": [0.62]}
    # algorithms.update({"Chinese Whisperers": (chinese_whispers_dlib, chinese_whisperers_range)})
    #
    # mean_shift_range = {"bandwidth": [0.54]}
    # algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
    #
    # dbscan_range = {"eps": [0.58], "min_samples": [1], "metric": ['euclidean']}
    # algorithms.update({"DBSCAN": (cluster_dbscan, dbscan_range)})

    # app_rank_order_range = {"threshold": np.arange(0.1, 0.4, 0.01), "n_neighbors": range(10, 1000, 10),
    #                         "distance": ["euclidean"]}
    # algorithms.update({"Approximate Rank-Order": (cluster_app_rank_order, app_rank_order_range)})
    #
    # affinity_range = {"damping": np.arange(0.6, 1, 0.02)}
    # algorithms.update({"Affinity Propagation": (cluster_affinity_propagation, affinity_range)})
    #
    # agglomerative_range = {"n_clusters": [None], "distance_threshold": np.arange(0.75, 1.25, 0.01)}
    # algorithms.update({"Agglomerative Clustering": (cluster_agglomerative, agglomerative_range)})
    #
    # optics_range = {"min_samples": range(1, 5), "eps": np.arange(0.7, 1.2, 0.01), "cluster_method": ["dbscan"],
    #                 "metric": ["euclidean"], "max_eps": [2]}
    # algorithms.update({"OPTICS": (cluster_optics, optics_range)})
    #
    # rank_order_range = {"threshold": range(1, 100, 5), "k_neighbors": range(10, 1000, 10),
    #                     "distance": ["euclidean"]}
    # algorithms.update({"Rank-Order": (cluster_rank_order, rank_order_range)})

    # kmeans_range = {"n_clusters": [5320]}
    # algorithms.update({"K-means": (cluster_kmeans, kmeans_range)})

    evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, n_threads=1,
                                   inter_logging=True)


def test_embeddings_creation_time():
    models = [FaceVGGSqueezeNet()]
    image = "./lfw/images/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"

    image = cv2.imread(image)
    for model in models:
        times = []
        for _ in trange(10000):
            begin = default_timer()
            processed = cv2.resize(image, model.InputSize, interpolation=cv2.INTER_CUBIC)
            processed = model.preprocess_input(processed)
            model.predict(processed)
            end = default_timer()
            times.append(end - begin)

        print(f"mean: {np.array(times).mean()} std: {np.array(times).std()}\n")


def test_small_subset():
    embedding_file = embeddings_dir + "/LFW/lfw-mtcnn-aligned/SqueezeNetVGG.txt"

    algorithms = dict()

    # threshold_range = {"threshold": np.arange(0.2, 1, 0.02), "distance": [find_euclidean_distance]}
    # algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

    chinese_whisperers_range = {"threshold": np.arange(0.2, 1, 0.01)}
    algorithms.update({"Chinese Whisperers": (chinese_whispers_dlib, chinese_whisperers_range)})
    #
    # mean_shift_range = {"bandwidth": np.arange(0.01, 1, 0.02)}
    # algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
    #
    # dbscan_range = {"eps": np.arange(0.2, 1, 0.02), "min_samples": [1], "metric": ['euclidean']}
    # algorithms.update({"DBSCAN": (cluster_dbscan, dbscan_range)})
    #
    # affinity_range = {"damping": np.arange(0.5, 1, 0.02)}
    # algorithms.update({"Affinity Propagation": (cluster_affinity_propagation, affinity_range)})

    app_rank_order_range = {"threshold": np.arange(0.1, 1, 0.02), "n_neighbors": range(10, 30, 2),
                            "distance": ["euclidean"]}
    algorithms.update({"Approximate Rank-Order": (cluster_app_rank_order, app_rank_order_range)})

    # agglomerative_range = {"n_clusters": [None], "distance_threshold": np.arange(1, 2, 0.02)}
    # algorithms.update({"Agglomerative Clustering": (cluster_agglomerative, agglomerative_range)})
    #
    # kmeans_range = {"n_clusters": range(20, 40, 2)}
    # algorithms.update({"K-means": (cluster_kmeans, kmeans_range)})
    #
    # spectral_range = {"n_clusters": range(20, 40, 2)}
    # algorithms.update({"Spectral Clustering": (cluster_spectral, spectral_range)})

    rank_order_range = {"threshold": np.arange(14, 16, 0.1), "k_neighbors": range(4, 13, 1),
                        "distance": ["euclidean"]}
    algorithms.update({"Rank-Order": (cluster_rank_order, rank_order_range)})

    evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, n_threads=1, top_n=100,
                                   inter_logging=False)


def test_clustering_algorithms(embedding_file):
    logger = get_file_logger()

    algorithms = dict()
    distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]

    threshold_range = {"threshold": [0], "distance": [find_euclidean_distance]}
    algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

    # chinese_whisperers_range = {"threshold": [1, 3, 5, 10], "iterations": [10], "distance": [find_euclidean_distance]}
    # algorithms.update({"Chinese Whisperers": (chinese_whisperers, chinese_whisperers_range)})
    #
    # rank_order_range = {"threshold": np.arange(0, 11, 0.1), "k_neighbors": range(7, 17),
    #                     "distance": ["euclidean", "manhattan"]}
    # algorithms.update({"Rank-Order": (cluster_rank_order, rank_order_range)})

    # mean_shift_range = {"bandwidth": np.arange(0.1, 10.0, 0.2), "n_jobs": [-1]}
    # algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
    #
    # dbscan_range = {"eps": range(1, 50), "min_samples": range(1, 8), "metric": distances}
    # algorithms.update({"DBSCAN": (cluster_dbscan, dbscan_range)})
    #
    # kmeans_range = {"n_clusters": range(1, 10), "random_state": range(50, 300, 10)}
    # algorithms.update({"K-means": (cluster_kmeans, kmeans_range)})
    #
    # affinity_range = {"damping": np.arange(0.5, 1, 0.1)}
    # algorithms.update({"Affinity Propagation": (cluster_affinity_propagation, affinity_range)})
    #
    # spectral_range = {"n_clusters": range(1, 10), "random_state": range(50, 300, 10)}
    # algorithms.update({"Spectral Clustering": (cluster_spectral, spectral_range)})
    #
    # agglomerative_range = {"n_clusters": [None], "distance_threshold": np.arange(0.01, 10, 0.01)}
    # algorithms.update({"Agglomerative Clustering": (cluster_agglomerative, agglomerative_range)})
    #
    # optics_range = {"min_samples": range(2, 10), "metric": distances}
    # algorithms.update({"OPTICS": (cluster_optics, optics_range)})

    evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, results_path=sorted_path)


def main():
    # elijah_test_1()
    # test_extraction_and_alignment()
    #
    # faces = "./results/extraction/LFW/lfw-mtcnn-aligned"
    # embeddings = "./results/embeddings/LFW/lfw-mtcnn-aligned"

    # test_embeddings_creation(faces, embeddings)

    test_small_subset()
    # test_embeddings_creation_time()

    # elijah_test_1()


if __name__ == "__main__":
    main()
