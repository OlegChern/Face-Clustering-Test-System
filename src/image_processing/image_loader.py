import os
import cv2

from timeit import default_timer


# Class for loading images from directories. Provides an image generator method.
class ImageLoader:

    def __init__(self, path, preproc_func=None, target_size=None, rgb=False):
        self.ImagePreprocessingTime = 0
        self.ImageResizeTime = 0
        self.ColorModeRGB = rgb

        self.ImageDirectoryPath = path
        self.PreprocessFunction = preproc_func
        self.TargetSize = target_size
        self.ImagesList = []

        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                self.ImagesList.append(file_path.replace("\\", "/"))

    # A generator method
    def next_image(self):
        for image_path in self.ImagesList:
            image = cv2.imread(image_path, 1)

            if self.ColorModeRGB:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.TargetSize is not None:
                start = default_timer()
                image = cv2.resize(image, self.TargetSize, interpolation=cv2.INTER_CUBIC)
                end = default_timer()

                self.ImageResizeTime += (end - start)

            if self.PreprocessFunction is not None:
                start = default_timer()
                image = self.PreprocessFunction(image)
                end = default_timer()

                self.ImagePreprocessingTime += (end - start)

            yield image, image_path

    def get_total_images_number(self):
        return len(self.ImagesList)
