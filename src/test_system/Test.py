from src.utils.ImageLoader import ImageLoader
from src.clustering.ImageClusterer import ImageClusterer
from src.clustering.Algorithm import optimal_params_grid_search
from src.utils.Utils import sort_images, evaluate_metrics
from timeit import default_timer

import os


class TestSystem:
    PathToResults = "./results/clustered"
    PathToEmbeddings = "./results/embeddings/embeddings.txt"

    def __init__(self, logger=None):
        self.Logger = logger

    def test_embeddings(self, embedder, path_to_faces, path_to_embeddings=PathToEmbeddings):
        loader = ImageLoader(path_to_faces, embedder.preprocess_image)

        start_time = default_timer()
        embedder.create_embeddings(loader, path_to_embeddings)
        end_time = default_timer()

        print(f"Embedding creation time is {end_time - start_time}s")

    def test_clustering(self, algorithms, embedding_path=PathToEmbeddings, results_path=PathToResults):
        clusterer = ImageClusterer(embedding_path)

        for algorithm in algorithms:
            start_time = default_timer()
            results = clusterer.cluster_images(algorithm)
            end_time = default_timer()

            path = results_path + f"/{algorithm.Name}"
            if not os.path.exists(path):
                os.mkdir(path)

            sort_images(results, path)
            prec, rec, f1 = evaluate_metrics(results)

            print(
                f"Algorithm {algorithm.Name} finished working in {end_time - start_time}s "
                f"with precision {prec}, recall {rec}, f1 {f1}")

    def test_clustering_with_optimal_params(selfs, algorithm, params_range, embedding_path=PathToEmbeddings):
        clusterer = ImageClusterer(embedding_path)

        best_prec, best_params_prec, best_rec, best_params_rec, best_f1, best_params_f1 = optimal_params_grid_search(
            clusterer, algorithm, params_range)

        print(
            f"For this set:\n"
            f"Best possible precision is {best_prec} with params {best_params_prec}\n"
            f"Best possible recall is {best_rec} with params {best_params_rec}\n"
            f"best possible f1 score is {best_f1} with params {best_params_f1}")
