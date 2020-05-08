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
from src.clustering.algorithms.algorithms import chinese_whisperers, cluster_threshold
from src.extraction.face_extraction import FaceExtractorDlib, FaceExtractorMTCNN

import numpy as np

embeddings_dir = "./results/embeddings"
save_path = "./results/extraction/"
load_path = "./results/extraction/Dlib-based Face Extractor/VGG-Face Cropping"
images_path = "./images"
sorted_path = "./results/clustered"

facenet_sandberg_weights = "./models/facenet_david_sandberg/facenet_weights.h5"
facenet_hiroki_weights = "./models/facenet_hiroki_taniai/facenet_hiroki_weights.h5"
openface_weights = "./models/open_face/openface_weights.h5"
vgg_face_weights = "./models/vgg_face/vgg_face_weights.h5"


def test_extraction_and_alignment():
    # logger = get_file_logger()

    extractors = list()
    extractors.append(FaceExtractorMTCNN())
    extractors.append(FaceExtractorDlib())

    normalizers = list()
    normalizers.append(EyesOnlyAligner())
    normalizers.append(EyesNoseAligner())
    normalizers.append(MappingAligner())
    normalizers.append(FaceCropperVGG())

    evaluate_normalizers(images_path, save_path, extractors, normalizers)


def test_embeddings_creation(faces_path):
    # logger = get_file_logger()

    models = list()
    # models.append((FaceNet, {"weights_path": facenet_sandberg_weights}))
    models.append((OpenFace, {"weights_path": openface_weights}))
    # models.append((FaceVGGSqueezeNet, {}))
    # models.append((FaceVGGResNet, {}))
    # models.append((FaceVGG16, {}))

    evaluate_embeddings_creator(models, faces_path, embeddings_dir)


def test_clustering_algorithms(embedding_file):
    # logger = get_file_logger()

    algorithms = dict()
    distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]

    # threshold_range = {"threshold": np.arange(0, 20, 0.01), "distance": distances}
    # algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

    # chinese_whisperers_range = {"threshold": np.arange(0, 10, 0.1), "iterations": [10, 20], "distance": distances}
    # algorithms.update({"Chinese Whisperers": (chinese_whisperers, chinese_whisperers_range)})
    #
    app_rank_order_range = {"threshold": np.arange(0, 10, 0.1), "n_neighbors": range(3, 11),
                            "distance": ["euclidean", "manhattan"]}
    algorithms.update({"Approximate Rank-Order": (cluster_app_rank_order, app_rank_order_range)})

    # rank_order_range = {"threshold": np.arange(0, 11, 0.1), "k_neighbors": range(7, 17),
    #                     "distance": ["euclidean", "manhattan"]}
    # algorithms.update({"Rank-Order": (cluster_rank_order, rank_order_range)})

    # mean_shift_range = {"bandwidth": np.arange(0.5, 1.0, 0.00001)}
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
    # faces_to_test = "./results/extraction/Dlib-based Face Extractor/Dlib Mapping Aligner"
    # test_embeddings_creation(faces_to_test)

    embeddings_file = embeddings_dir + "/OpenFace.txt"
    test_clustering_algorithms(embeddings_file)


if __name__ == "__main__":
    main()
