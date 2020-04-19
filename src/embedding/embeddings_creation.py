from src.image_processing.image_loader import ImageLoader


class ImageEmbeddingsCreator:
    DefaultEmbeddingsPath = "./results/embeddings/embeddings.txt"

    def __init__(self, faces_path):
        self.FacesPath = faces_path.replace("\\", "/")

    def create_embeddings(self, model, save_path=DefaultEmbeddingsPath):
        save_path = save_path.replace("\\", "/")
        loader = ImageLoader(self.FacesPath)

        with open(save_path, "w") as file:
            for image, image_path in loader.next_image():
                samples = model.preprocess_input(image)
                result = model.predict(samples)

                embedding = str(result[0, :])
                embedding = embedding.replace("\n", "")
                embedding = embedding.replace("[", "")
                embedding = embedding.replace("]", "")

                result_string = f"{image_path}\t{embedding}\n"
                file.write(result_string)
