from keras_vggface import VGGFace
from keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess_input
from src.embedding.embeddings_creation import AbstractEmbeddingModel

import cv2
import numpy as np


class FaceVGG(AbstractEmbeddingModel):
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
        return self.Model.predict(x)


class FaceVGG16(FaceVGG):
    def __init__(self):
        super(FaceVGG16, self).__init__(model="vgg16", name="VGG16")


class FaceVGGResNet(FaceVGG):
    def __init__(self):
        super(FaceVGGResNet, self).__init__(model="resnet50", name="ResNetVGG")


class FaceVGGSqueezeNet(FaceVGG):
    def __init__(self):
        super(FaceVGGSqueezeNet, self).__init__(model="senet50", name="SqueezeNetVGG")