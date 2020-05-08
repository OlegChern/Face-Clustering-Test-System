from src.image_processing.image_loader import ImageLoader
from timeit import default_timer
from abc import abstractmethod

from numpy import array_str
import numpy as np
import sys


class AbstractEmbeddingModel:
    InputShape = None
    InputSize = None

    @abstractmethod
    def preprocess_input(self, image):
        ...

    @abstractmethod
    def predict(self, x):
        ...


class ImageEmbeddingsCreator:
    DefaultEmbeddingsPath = "./results/embeddings/embeddings.txt"
    EmbeddingsCreationTime = 0
    ImagePreprocessingTime = 0

    def __init__(self, faces_path):
        self.FacesPath = faces_path.replace("\\", "/")

    def create_embeddings(self, model, save_path=DefaultEmbeddingsPath):
        self.EmbeddingsCreationTime = 0
        self.ImagePreprocessingTime = 0

        save_path = save_path.replace("\\", "/")
        loader = ImageLoader(self.FacesPath)

        np.set_printoptions(threshold=sys.maxsize, precision=15)
        with open(save_path, "w") as file:
            for image, image_path in loader.next_image():
                start = default_timer()
                samples = model.preprocess_input(image)
                end = default_timer()

                self.ImagePreprocessingTime += (end - start)

                start = default_timer()
                result = model.predict(samples)
                end = default_timer()

                self.EmbeddingsCreationTime += (end - start)

                embedding = array_str(result[0, :], precision=15)
                embedding = embedding.replace("\n", "")
                embedding = embedding.replace("[", "")
                embedding = embedding.replace("]", "")

                result_string = f"{image_path}\t{embedding}\n"
                file.write(result_string)
