from src.embedding.models_code.open_face import OpenFace
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

from src.image_processing.utils import find_cosine_similarity, find_euclidean_distance

import numpy as np


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    euclidean_distance = l2_normalize(euclidean_distance )
    return euclidean_distance


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(96, 96))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img.astype("float32")
    # img = preprocess_input(img)
    return img


openface_weights = "../models_code/open_face/openface_weights.h5"

model = OpenFace(openface_weights)

cage_1 = "../results/extraction/non_aligned/cage_2/cage_2_face_1.jpg"
cage_2 = "../results/extraction/non_aligned/cage_6/cage_6_face_0.jpg"
saoirse_1 = "../results/extraction/non_aligned/saoirse_1/saoirse_1_face_0.jpg"
saoirse_2 = "../results/extraction/non_aligned/saoirse_4/saoirse_4_face_0.jpg"

cage_1_vector = model.predict(preprocess_image(cage_1))[0, :]
cage_2_vector = model.predict(preprocess_image(cage_2))[0, :]

saoirse_1_vector = model.predict(preprocess_image(saoirse_1))[0, :]
saoirse_2_vector = model.predict(preprocess_image(saoirse_2))[0, :]

print(np.dot(cage_1_vector - cage_2_vector, cage_1_vector - cage_2_vector))
print(np.dot(saoirse_1_vector - cage_2_vector, saoirse_1_vector - cage_2_vector))
print(np.dot(saoirse_1_vector - saoirse_2_vector, saoirse_1_vector - saoirse_2_vector))
