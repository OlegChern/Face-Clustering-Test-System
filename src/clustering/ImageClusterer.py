import numpy as np


# An abstract class for future implementations of face clustering logic
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

    def cluster_images(self, algorithm):
        vectors = np.asarray(self.Vectors)
        labels = algorithm.fit_predict(vectors)

        return zip(self.Paths, labels)
