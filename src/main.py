from src.clustering.algorithms.approximate_rank_order import cluster_app_rank_order
from src.clustering.algorithms.rank_order import cluster_rank_order
from src.clustering.algorithms.scikit_algorithms import cluster_mean_shift, cluster_dbscan, cluster_kmeans, \
    cluster_affinity_propagation, cluster_spectral, cluster_agglomerative, cluster_optics
from src.image_processing.image_loader import ImageLoader
from src.clustering.clustering_utils import find_taxicab_distance, find_cosine_similarity, find_euclidean_distance
from src.embedding.models_code.face_net import FaceNet
from src.embedding.models_code.open_face import OpenFace
from src.embedding.models_code.deep_face import DeepFace
from src.embedding.models_code.vgg_face import FaceVGG
from src.test_system.evaluation import evaluate_normalizers, evaluate_embeddings_creator, \
    evaluate_clustering_algorithms
from src.extraction.face_normalization import EyesNoseAligner, EyesOnlyAligner, MappingAligner, FaceCropperVGG
from src.clustering.algorithms.algorithms import chinese_whisperers, cluster_threshold
from src.extraction.face_extraction import FaceExtractorDlib, FaceExtractorMTCNN

import numpy as np

embedding_file = "./results/embeddings/deepface_embeddings.txt"
save_path = "./results/extraction/"
load_path = "./results/extraction/Dlib_alignment_224"
images_path = "./images"
sorted_path = "./results/clustered"

facenet_sandberg_model = "./models/facenet_david_sandberg/facenet_weights.h5"
facenet_hiroki_weights = "./models/facenet_hiroki_taniai/facenet_hiroki_weights.h5"
openface_weights = "./models/open_face/openface_weights.h5"
vgg_face_weights = "./models/vgg_face/vgg_face_weights.h5"
deep_face_weigths = "./models/deep_face/VGGFace2_DeepFace_weights_val-0.9034.h5"


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

test_extraction_and_alignment()


# test_extraction_and_alignment()

models = list()
#
# facenet_file = "./results/embeddings/facenet_embeddings.txt"
# facenet_model = (FaceNet, facenet_sandberg_model, facenet_file)
#
# models.append(facenet_model)
#
#
#
# extractor = FaceExtractorDlib()
# loader = ImageLoader(images_path)
# extractor.extract_faces(loader, load_path, align=True)

# openface_file = "./results/embeddings/openface_embeddings.txt"
# openface_model = (OpenFace, openface_weights, openface_file)
# models.append(openface_model)

# vgg_file = "./results/embeddings/vgg_embeddings.txt"
# vgg_model = (FaceVGG, vgg_face_weights, vgg_file)
# models.append(vgg_model)
#
#
deepface_model = DeepFace(deep_face_weigths)
deepface_file = "./results/embeddings/deepface_embeddings.txt"
models.append((DeepFace, deep_face_weigths, deepface_file))
#
# evaluate_embeddings_creator(models, load_path)

algorithms = dict()
distances = [find_euclidean_distance, find_cosine_similarity, find_taxicab_distance]

# app_rank_order_range = {"threshold": np.arange(0, 200, 0.2), "n_neighbors": [5]}
# algorithms.update({"Approximate Rank-Order": (cluster_app_rank_order, app_rank_order_range)})
#
# rank_order_range = {"threshold": np.arange(0, 200, 0.2), "k_neighbors": range(5, 11), "distance": ["euclidean", "manhattan"]}
# algorithms.update({"Rank-Order": (cluster_rank_order, rank_order_range)})
#
#
# threshold_range = {"threshold": np.arange(0, 200, 0.1), "distance": distances}
# algorithms.update({"Threshold Clustering": (cluster_threshold, threshold_range)})
#
# chinese_whisperers_range = {"threshold": np.arange(0, 200, 0.1), "iterations": [10],
#                             "distance": distances}
# algorithms.update({"Chinese Whisperers": (chinese_whisperers, chinese_whisperers_range)})

mean_shift_range = {"bandwidth": np.arange(0.5, 1.0, 0.00001)}
algorithms.update({"Mean Shift": (cluster_mean_shift, mean_shift_range)})
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

# evaluate_clustering_algorithms(algorithms, embedding_path=embedding_file, results_path=sorted_path)
