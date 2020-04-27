import numpy as np

from src.image_processing.utils import l2_normalize


class ImageClusteringUnit:
    Vectors = []
    Paths = []

    def __init__(self, embedding_path, normalize=False):
        with open(embedding_path, "r") as embeddings:
            for line in embeddings.readlines():
                path, vector_str = line.split("\t")
                vector = np.fromstring(vector_str, dtype="float32", sep=" ")

                self.Paths.append(path)
                self.Vectors.append(vector)

        self.Vectors = np.asarray(self.Vectors)
        if normalize:
            self.Vectors = l2_normalize(self.Vectors)

    def cluster_images(self, algorithm, params_dict=None):
        labels = algorithm(self.Vectors, params_dict)

        return list(zip(self.Paths, labels))
