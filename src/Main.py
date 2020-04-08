from src.test_system.Test import TestSystem
from src.utils.ImageLoader import ImageLoader
from src.extraction.FaceExtractor import FaceExtractorMTCNN
from src.clustering.Algorithm import DbscanAlgorithm, KmeansAlgorithm, MeanShiftAlgorithm
from src.embedding.EmbeddingCreator import FaceNetEmbedder

test_file = "./results/embeddings/embeddings.txt"
save_path = "./results/extraction"
images_path = "./images"

# extractor = FaceExtractorMTCNN()
# loader = ImageLoader(images_path)
# extractor.extract_faces(loader, save_path)

test_system = TestSystem()

algorithms = [MeanShiftAlgorithm({"bandwidth": 11})]
test_system.test_clustering(algorithms)

# params_range = {"bandwidth": range(1, 101, 1)}
#test_system.test_clustering_with_optimal_params(MeanShiftAlgorithm, params_range)
