import cv2

from tensorflow import keras
from numpy import asarray, expand_dims
from src.utils.utils import l2_normalize
from src.embedding.inception_resnet_v1 import InceptionResNetV1
from keras import models

import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

import json
from keras.models import load_model
from keras.models import model_from_json

facenet_taniai_model = "./models/facenet_hiroki_taniai/facenet_keras.h5"
facenet_sandberg_model = "./models/facenet_david_sandberg/facenet_weights.h5"
facenet_sandberg_structure = r"C:\Users\Olegator\Desktop\Course Work\Face-Clustering-Test-System\models\facenet_david_sandberg\facenet_model.json"


def facenet_preprocess_image(image):
    image = cv2.resize(image, (160, 160))

    pixels = asarray(image)
    pixels = pixels.astype('float32')

    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std

    samples = expand_dims(pixels, axis=0)

    return samples


def test_preprocess(image):
    img = cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)
    pixels = img_to_array(img)
    # pixels = pixels[:, :, ::-1]

    pixels = np.expand_dims(pixels, axis=0)
    pixels = pixels.astype('float32')

    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std

    return pixels


# https://github.com/nyoki-mtl/keras-facenet
def load_taniai_model(model_path=facenet_taniai_model):
    model = keras.models.load_model(model_path)
    return model


# https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
def load_sandberg_model():
    model = InceptionResNetV1()
    model.load_weights(facenet_sandberg_model)
    return model


def facenet_create_embeddings(loader, save_path, model):
    save_path = save_path.replace("\\", "/")

    with open(save_path, "w") as file:
        for image, image_path in loader.next_image():
            samples = test_preprocess(image)
            result = model.predict(samples)

            embedding = l2_normalize(result[0, :])
            embedding = str(embedding)
            embedding = embedding.replace("\n", "")
            embedding = embedding.replace("[", "")
            embedding = embedding.replace("]", "")

            result_string = f"{image_path}\t{embedding}\n"
            file.write(result_string)
