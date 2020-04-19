from src.test_system.evaluation import evaluate_embeddings_creator, evaluate_clustering_algorithms
from src.embedding.models.open_face import OpenFace
from src.embedding.models.face_net import FaceNet
from src.clustering.algorithms.algorithms import chinese_whisperers
from src.clustering.algorithms.utils import find_taxicab_distance, find_cosine_similarity, find_euclidean_distance
from src.test_system.logging import get_file_logger

import numpy as np

embedding_file = "./results/embeddings/embeddings.txt"
save_path = "./results/extraction"
images_path = "./images"
sorted_path = "./results/clustered"

facenet_sandberg_model = "./models/facenet_david_sandberg/facenet_weights.h5"
facenet_hiroki_weights = "./models/facenet_hiroki_taniai/facenet_hiroki_weights.h5"
openface_weights = "./models/open_face/openface_weights.h5"

# logger = Logger()
#
# logger.info("Test session started")

# loader = ImageLoader(images_path)
# extract_faces_mtcnn(loader, save_path)

# evaluate_facenet(load_sandberg_model(), save_path)
# evaluate_facenet(load_taniai_model(), save_path)

# model, image_size = load_openface_model()
# model, image_size = load_sandberg_model()

logger = get_file_logger()

facenet_model = FaceNet(facenet_hiroki_weights)
facenet_file = "./results/embeddings/embeddings.txt"
models = {facenet_model: facenet_file}
# model = OpenFace(openface_weights)
evaluate_embeddings_creator(models, save_path, logger=logger)

algorithms = dict()
distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]
#
# threshold_range = {"threshold": np.arange(0, 10, 0.01), "distance": distances}
# algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

chinese_whisperers_range = {"threshold": np.arange(0.01, 1, 0.01), "iterations": range(10, 11),
                            "distance": [find_cosine_similarity]}
algorithms.update({"Chinese Whisperers": (chinese_whisperers, chinese_whisperers_range)})
#
# mean_shift_range = {"bandwidth": np.arange(0.1, 40, 0.1)}
# algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
# #
# dbscan_range = {"eps": range(1, 30), "min_samples": range(1, 5), "metric": distances}
# algorithms.update({"DBSCAN": (cluster_dbscan, dbscan_range)})
#
# kmeans_range = {"n_clusters": range(1, 10), "random_state": range(50, 300, 10)}
# algorithms.update({"K-means": (cluster_kmeans, kmeans_range)})

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


evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, results_path=sorted_path, logger=logger)
