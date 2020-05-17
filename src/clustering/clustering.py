import numpy as np

from src.clustering.clustering_utils import l2_normalize


class ImageClusteringUnit:

    def __init__(self, embedding_path, normalize=True):
        self.Paths = list()
        self.Vectors = list()

        with open(embedding_path, "r") as embeddings:
            for line in embeddings.readlines():
                path, vector_str = line.split("\t")
                vector = np.fromstring(vector_str, dtype="float32", sep=" ")

                self.Paths.append(path)
                self.Vectors.append(vector)

        self.Vectors = np.asarray(self.Vectors)
        if normalize:
            self.Vectors = l2_normalize(self.Vectors)

    def get_total_vectors_number(self):
        return len(self.Vectors)

    def cluster_images(self, algorithm, params_dict=None, top_n=None):
        if top_n is None:
            labels = algorithm(self.Vectors, params_dict)
        else:
            labels = algorithm(self.Vectors[:top_n], params_dict)

        return list(zip(self.Paths, labels))
