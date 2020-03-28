import os
from src.clustering.ImageClusterer import ImageClusterer
from src.clustering.Algorithm import DbscanAlgorithm
from src.utils.ImageLoader import ImageLoader
from src.utils.Utils import sort_images
from src.extraction.FaceExtractor import FaceExtractorMTCNN
from src.embedding.EmbeddingCreator import FaceNetEmbedder, preprocess_image_facenet

# image_path = os.path.join("C:\\Users", "Olegator", "Desktop", "reunion")
# save_path = "C:\\Users\\Olegator\\Desktop\\Course Work\\Face-Clustering-Test-System\\results\\extraction"

# loader = ImageLoader(image_path)
# extractor = FaceExtractorMTCNN()
#
# extractor.extract_faces(loader, save_path)

test_file = "./results/embeddings/test.txt"
# model_path = "./models/facenet_keras.h5"
# faces_path = "./results/extraction"
#
# embedder = FaceNetEmbedder(model_path)
# loader = ImageLoader(faces_path, preprocess_image_facenet)
#
# embedder.create_embeddings(loader, test_file)


cluster_test = ImageClusterer(test_file)
algorithm_test = DbscanAlgorithm()

result = cluster_test.cluster_images(algorithm_test)

save_path = "./results/clustered"
sort_images(result, save_path)