from src.extraction.mtcnn_face_extraction import extract_faces_mtcnn
from src.image_processing.image_loader import ImageLoader
from src.test_system.evaluation import evaluate_embeddings_creator, evaluate_clustering_algorithms
from src.embedding.models_code.face_net import FaceNet
from src.embedding.models_code.open_face import OpenFace
from src.embedding.models_code.vgg_face import FaceVGG
from src.embedding.models_code.deep_face import DeepFace
from src.clustering.algorithms.algorithms import chinese_whisperers, cluster_threshold
from src.clustering.algorithms.scikit_algorithms import cluster_optics, cluster_agglomerative, cluster_spectral, \
    cluster_affinity_propagation, cluster_dbscan, cluster_mean_shift, cluster_kmeans
from src.image_processing.utils import find_taxicab_distance, find_cosine_similarity, find_euclidean_distance
from src.test_system.logging import get_file_logger

import numpy as np

embedding_file = "./results/embeddings/embeddings.txt"
save_path = "./results/extraction/non_aligned"
images_path = "./images"
sorted_path = "./results/clustered"

facenet_sandberg_model = "./models_code/facenet_david_sandberg/facenet_weights.h5"
facenet_hiroki_weights = "./models_code/facenet_hiroki_taniai/facenet_hiroki_weights.h5"
openface_weights = "./models_code/open_face/openface_weights.h5"
vgg_face_weights = "./models_code/vgg_face/vgg_face_weights.h5"
deep_face_weigths = "./models_code/deep_face/VGGFace2_DeepFace_weights_val-0.9034.h5"

# loader = ImageLoader(images_path)
# extract_faces_mtcnn(loader, save_path, align=True)

# logger = get_file_logger()

models = dict()

facenet_model = FaceNet(facenet_sandberg_model)
facenet_file = "./results/embeddings/embeddings.txt"
models.update({facenet_model: facenet_file})

# openface_model = OpenFace(openface_weights)
# openface_file = "./results/embeddings/embeddings.txt"
# models_code.update({openface_model: openface_file})

# vgg_model = FaceVGG(vgg_face_weights)
# vgg_file = "./results/embeddings/embeddings.txt"
# models_code.update({vgg_model: vgg_file})
#

# deepface_model = DeepFace(deep_face_weigths)
# deepface_file = "./results/embeddings/embeddings.txt"
# models_code.update({deepface_model: deepface_file})
#
evaluate_embeddings_creator(models, save_path)

algorithms = dict()
distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]

threshold_range = {"threshold": np.arange(0.1, 40, 0.01), "distance": distances}
algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})

chinese_whisperers_range = {"threshold": np.arange(0.1, 40, 0.1), "iterations": range(10, 11),
                            "distance": distances}
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
#
evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, results_path=sorted_path)
