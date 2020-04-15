from src.image_processing.image_loader import ImageLoader
from src.extraction.mtcnn_face_extraction import extract_faces_mtcnn
from src.test_system.evaluation import evaluate_facenet, evaluate_clustering_algorithms, \
    evaluate_clustering_algorithm_with_optimal_params
from src.embedding.embeddings_creation import load_sandberg_model, load_taniai_model
from src.clustering.algorithms import cluster_threshold, \
    chinese_whisperers
from src.clustering.scikit_algorithms import cluster_kmeans, cluster_dbscan, cluster_mean_shift, \
    cluster_affinity_propagation, cluster_spectral
from src.test_system.logging import Logger
from src.clustering.utils import find_taxicab_distance, find_cosine_similarity, find_euclidean_distance

import numpy as np

embedding_file = "../results/embeddings/embeddings.txt"
save_path = "../results/extraction"
images_path = "../images"

# logger = Logger()
#
# logger.info("Test session started")

# loader = ImageLoader(images_path)
# extract_faces_mtcnn(loader, save_path)

# evaluate_facenet(load_sandberg_model(), save_path)
# evaluate_facenet(load_taniai_model(), save_path)

algorithms = dict()
#
# threshold_range = {"threshold": np.arange(0, 10, 0.01)}
# algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})
#
distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]
#

# chinese_whisperers_range = {"threshold": np.arange(0.01, 10, 0.01), "iterations": range(10, 11),
#                             "distance": distances}
# algorithms.update({"Chinese Whisperers": (chinese_whisperers, chinese_whisperers_range)})

# mean_shift_range = {"bandwidth": np.arange(0.1, 40, 0.1)}
# algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
#
dbscan_range = {"eps": range(1, 30), "min_samples": range(1, 5), "metric": distances}
algorithms.update({"DBSCAN": (cluster_dbscan, dbscan_range)})
#
# kmeans_range = {"n_clusters": range(1, 10), "random_state": range(50, 300, 10)}
# algorithms.update({"K-means": (cluster_kmeans, kmeans_range)})

# affinity_range = {"damping": np.arange(0.5, 1, 0.1)}
# algorithms.update({"Affinity Propagation": (cluster_affinity_propagation, affinity_range)})

spectral_range = {"n_clusters": range(1, 10), "random_state": range(50, 300, 10)}
algorithms.update({"Spectral Clustering": (cluster_spectral, spectral_range)})

evaluate_clustering_algorithms(algorithms)
