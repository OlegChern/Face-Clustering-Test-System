import cv2
from tensorflow import keras
from numpy import asarray, expand_dims


def preprocess_image_facenet(image):
    image = cv2.resize(image, (160, 160))
    return image


# An interface for future implementations of face embedding logic
class EmbeddingCreator:
    def create_embeddings(self, loader, save_path):
        pass


class FaceNetEmbedder(EmbeddingCreator):
    Model = None

    def __init__(self, model_path):
        self.Model = keras.models.load_model(model_path)

    def create_embeddings(self, loader, save_path):
        save_path = save_path.replace("\\", "/")
        with open(save_path, "w") as file:
            for image, image_path in loader.next_image():
                pixels = asarray(image)
                pixels = pixels.astype('float32')

                mean, std = pixels.mean(), pixels.std()
                pixels = (pixels - mean) / std

                samples = expand_dims(pixels, axis=0)
                result = self.Model.predict(samples)

                embedding = str(result[0])
                embedding = embedding.replace("\n", "")
                embedding = embedding.replace("[", "")
                embedding = embedding.replace("]", "")

                result_string = f"{image_path}\t{embedding}\n"
                file.write(result_string)
