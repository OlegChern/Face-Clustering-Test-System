import os
import cv2


# Class for loading images from directories. Provides an image generator method.
class ImageLoader:

    def __init__(self, path, preproc_func=None, target_size=None):
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
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.TargetSize is not None:
                image = cv2.resize(image, self.TargetSize)

            if self.PreprocessFunction is not None:
                image = self.PreprocessFunction(image)

            yield image, image_path
