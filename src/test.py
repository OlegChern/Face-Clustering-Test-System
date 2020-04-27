from src.clustering.clustering import ImageClusteringUnit
from src.clustering.algorithms.approximate_rank_order import cluster_app_rank_order


facenet_file = "../results/embeddings/embeddings.txt"

unit = ImageClusteringUnit(facenet_file)
results = unit.cluster_images(cluster_app_rank_order, params_dict={"threshold": [5], "n_neighbors": 5})

print(results)