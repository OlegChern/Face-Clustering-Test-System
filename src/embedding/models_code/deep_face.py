import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Input, concatenate, Convolution2D, Dropout, LocallyConnected2D
from keras.layers.core import Dense, Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.applications.imagenet_utils import preprocess_input as preprocess_imagenet
from keras.preprocessing.image import img_to_array

import cv2
import numpy as np


class DeepFace:
    InputSize = (152, 152)
    InputShape = (152, 152, 3)

    def __init__(self, weights_path, name="DeepFace"):
        self.Name = name

        self.Model = self.create_model()
        self.Model.load_weights(weights_path)
        self.truncate_model()

    def preprocess_input(self, image):
        image = cv2.resize(image, self.InputSize)
        pixels = img_to_array(image)

        samples = np.expand_dims(pixels, axis=0)
        samples /= 255

        return samples

    def predict(self, x):
        return self.Model.predict(x)

    def truncate_model(self):
        self.Model = Model(inputs=self.Model.layers[0].input, outputs=self.Model.layers[-3].output)

    def create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=self.InputShape))
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
        model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
        model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
        model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
        model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
        model.add(Flatten(name='F0'))
        model.add(Dense(4096, activation='relu', name='F7'))
        model.add(Dropout(rate=0.5, name='D0'))
        model.add(Dense(8631, activation='softmax', name='F8'))

        return model
