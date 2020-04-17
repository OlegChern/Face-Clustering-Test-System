import cv2
import numpy as np

from tensorflow import keras
from numpy import asarray, expand_dims
from src.embedding.inception_resnet_v1 import InceptionResNetV1
from src.embedding.open_face import OpenFace
from keras.applications.imagenet_utils import preprocess_input


facenet_taniai_model = "./models/facenet_hiroki_taniai/facenet_keras.h5"
facenet_sandberg_model = "./models/facenet_david_sandberg/facenet_weights.h5"
openface_model = "./models/open_face/openface_weights.h5"


def preprocess_image(image, target_size=(160, 160)):
    image = cv2.resize(image, target_size)

    pixels = asarray(image)
    # pixels = pixels[:, :, ::-1]

    pixels = np.around(np.transpose(pixels, (1, 0, 2)) / 255.0, decimals=12)
    # pixels = pixels.astype('float32')
    # mean, std = pixels.mean(), pixels.std()
    # pixels = (pixels - mean) / std

    samples = expand_dims(pixels, axis=0)
    # samples = preprocess_input(samples)

    return samples


# https://github.com/nyoki-mtl/keras-facenet
def load_taniai_model(model_path=facenet_taniai_model):
    model = keras.models.load_model(model_path)
    image_size = (160, 160)

    return model, image_size


# https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
def load_sandberg_model():
    model = InceptionResNetV1()
    model.load_weights(facenet_sandberg_model)
    image_size = (160, 160)

    return model, image_size


# https://github.com/serengil/tensorflow-101/blob/master/model/openface_model.py
def load_openface_model():
    model = OpenFace()
    model.load_weights(openface_model)
    image_size = (96, 96)

    return model, image_size


def create_embeddings(loader, save_path, model, target_image_size=(160, 160)):
    save_path = save_path.replace("\\", "/")

    with open(save_path, "w") as file:
        for image, image_path in loader.next_image():
            samples = preprocess_image(image, target_image_size)
            result = model.predict(samples)

            embedding = str(result[0, :])
            embedding = embedding.replace("\n", "")
            embedding = embedding.replace("[", "")
            embedding = embedding.replace("]", "")

            result_string = f"{image_path}\t{embedding}\n"
            file.write(result_string)
