from keras_vggface import VGGFace
from keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess_input

import cv2
import numpy as np


class FaceVGG:
    InputSize = (224, 224)
    InputShape = (224, 224, 3)

    def __init__(self, model, name="VGG-Face"):
        self.Name = name
        self.Model = VGGFace(include_top=False, model=model, input_shape=self.InputShape, pooling='avg')

    def preprocess_input(self, image):
        image = cv2.resize(image, self.InputSize, interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        image = imagenet_preprocess_input(image)

        return image

    def predict(self, x):
        return self.Model.predict(x)[0]

    def create_model(self):
        pass


class FaceVGG16(FaceVGG):
    def __init__(self):
        super(FaceVGG16, self).__init__(model="vgg16", name="VGG16-Face")


class FaceVGGResNet(FaceVGG):
    def __init__(self):
        super(FaceVGGResNet, self).__init__(model="resnet50", name="ResNet based VGG-Face")


class FaceVGGSqueezeNet(FaceVGG):
    def __init__(self):
        super(FaceVGGSqueezeNet, self).__init__(model="senet50", name="SqueezeNet based VGG-Face")

# class FaceVGG:
#     InputSize = (224, 224)
#     InputShape = (224, 224, 3)
#
#     def __init__(self, weights_path, name="VGG-Face"):
#         self.Name = name
#
#         self.Model = self.create_model()
#         self.Model.load_weights(weights_path)
#         self.truncate_model()
#
#     def preprocess_input(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, self.InputSize)
#
#         # pixels = np.asarray(image).astype(np.float32)
#
#         samples = np.expand_dims(image, axis=0)
#         samples = preprocess_imagenet(samples)
#
#         return samples
#
#     def predict(self, x):
#         return self.Model.predict(x)
#
#     def truncate_model(self):
#         self.Model = Model(inputs=self.Model.layers[0].input, outputs=self.Model.layers[-2].output)
#
#     def create_model(self):
#         model = Sequential()
#         model.add(ZeroPadding2D((1, 1), input_shape=self.InputShape))
#         model.add(Convolution2D(64, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(64, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(128, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(128, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(256, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(ZeroPadding2D((1, 1)))
#         model.add(Convolution2D(512, (3, 3), activation='relu'))
#         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#         model.add(Convolution2D(4096, (7, 7), activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Convolution2D(4096, (1, 1), activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Convolution2D(2622, (1, 1)))
#         model.add(Flatten())
#         model.add(Activation('softmax'))
#
#         return model
