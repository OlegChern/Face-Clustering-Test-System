from src.image_processing.image_loader import ImageLoader
from timeit import default_timer
from abc import abstractmethod
from tqdm import tqdm

from numpy import array_str
import numpy as np
import sys


class AbstractEmbeddingModel:
    InputShape = None
    InputSize = None

    @staticmethod
    @abstractmethod
    def preprocess_input(image):
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
        loader = ImageLoader(self.FacesPath, preproc_func=model.preprocess_input, target_size=model.InputSize)

        np.set_printoptions(threshold=sys.maxsize, precision=15)
        with open(save_path, "w") as file:
            progress_bar = tqdm(loader.next_image(), total=loader.get_total_images_number(),
                                desc=model.Name, leave=True)

            for image, image_path in progress_bar:
                start = default_timer()
                result = model.predict(image)
                end = default_timer()

                self.EmbeddingsCreationTime += (end - start)

                embedding = array_str(result[0, :], precision=15)
                embedding = embedding.replace("\n", "")
                embedding = embedding.replace("[", "")
                embedding = embedding.replace("]", "")

                result_string = f"{image_path}\t{embedding}\n"
                file.write(result_string)

        self.ImagePreprocessingTime = loader.ImagePreprocessingTime + loader.ImageResizeTime
