import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Input, concatenate, Convolution2D, Dropout
from keras.layers.core import Dense, Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.applications.imagenet_utils import preprocess_input as preprocess_imagenet

import cv2
import numpy as np


class FaceVGG:
    InputSize = (224, 224)
    InputShape = (224, 224, 3)

    def __init__(self, weights_path, name="VGG-Face"):
        self.Name = name

        self.Model = self.create_model()
        self.Model.load_weights(weights_path)
        self.truncate_model()

    def preprocess_input(self, image):
        image = image[:, :, ::-1]
        image = cv2.resize(image, self.InputSize)
        pixels = np.asarray(image)

        samples = np.expand_dims(pixels, axis=0)
        samples = preprocess_imagenet(samples)

        return samples

    def predict(self, x):
        return self.Model.predict(x)

    def truncate_model(self):
        self.Model = Model(inputs=self.Model.layers[0].input, outputs=self.Model.layers[-2].output)

    def create_model(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=self.InputShape))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model
