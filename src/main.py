from src.utils.image_loader import ImageLoader
from src.extraction.mtcnn_face_extraction import extract_faces_mtcnn
from src.test_system.evaluation import evaluate_embeddings_creator, evaluate_clustering_algorithms, \
    evaluate_clustering_algorithm_with_optimal_params
from src.embedding.embeddings_creation import facenet_create_embeddings
from src.clustering.algorithms import cluster_kmeans, cluster_mean_shift, cluster_dbscan, cluster_threshold
from src.test_system.logging import Logger

import numpy as np

embedding_file = "./results/embeddings/embeddings.txt"
save_path = "./results/extraction"
images_path = "./images"

logger = Logger()

# logger.info("Testing Logger")

# loader = ImageLoader(images_path)
# extract_faces_mtcnn(loader, save_path)

# evaluate_embeddings_creator(facenet_create_embeddings, save_path)

algorithms = {"Mean Shift": cluster_mean_shift, "K-means": cluster_kmeans, "DBSCAN": cluster_dbscan,
              "Threshold Clustering": cluster_threshold}
evaluate_clustering_algorithms(algorithms)

# params_dict = {"threshold": np.arange(0, 20, 0.01)}
# evaluate_clustering_algorithm_with_optimal_params(cluster_threshold, params_dict)
