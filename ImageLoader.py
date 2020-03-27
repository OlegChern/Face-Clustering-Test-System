import os
import cv2


# An abstract class for future implementations of image loading logic
class ImageLoader:
    ImageDirectoryPath = ''
    ImagesList = []

    def __init__(self, path):
        self.ImageDirectoryPath = path

        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                self.ImagesList.append(file_path.replace("\\", "/"))

    # A generator function
    def next_image(self):
        pass


# Details on MTCNN detector can be found here: https://pypi.org/project/mtcnn/
class ImageLoaderMTCNN(ImageLoader):
    def next_image(self):
        for image_path in self.ImagesList:
            image = cv2.imread(image_path)

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")

            yield image, image_name


class ImageLoaderFaceNet(ImageLoader):
    def next_image(self):
        for image_path in self.ImagesList:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (160, 160))

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")

            yield image, image_name
