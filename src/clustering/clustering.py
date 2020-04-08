import numpy as np


class ImageClusterer:
    Paths = []
    Vectors = []

    def __init__(self, embedding_path):
        with open(embedding_path, "r") as embeddings:
            for line in embeddings.readlines():
                path, vector_str = line.split("\t")
                vector = np.fromstring(vector_str, dtype="float32", sep=" ")

                self.Paths.append(path)
                self.Vectors.append(vector)

    def cluster_images(self, algorithm, params_dict=None):
        vectors = np.asarray(self.Vectors)
        labels = algorithm(vectors, params_dict)

        return list(zip(self.Paths, labels))
