import os
import cv2


def identity(image):
    return image


# Class for loading images from directories. Provides an image generator.
class ImageLoader:
    ImageDirectoryPath = ''
    ImagesList = []

    def __init__(self, path, preproc_func=identity):
        self.ImageDirectoryPath = path
        self.PreprocessFunc = preproc_func

        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                self.ImagesList.append(file_path.replace("\\", "/"))

    # A generator function
    def next_image(self):
        for image_path in self.ImagesList:
            image = cv2.imread(image_path)
            image = self.PreprocessFunc(image)

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")

            yield image, image_name
