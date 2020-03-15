from ImageLoader import ImageLoaderMTCNN
import os
from FaceExtractor import FaceExtractorMTCNN

image_path = os.path.join("C:\\Users", "Olegator", "Desktop", "reunion")
save_path = "C:\\Users\\Olegator\\Desktop\\Course Work\\Face-Clustering-Test-System\\results"

loader = ImageLoaderMTCNN(image_path)
test = loader.next_image().__next__()
print(test)
extractor = FaceExtractorMTCNN()

extractor.extract_faces(loader, save_path)