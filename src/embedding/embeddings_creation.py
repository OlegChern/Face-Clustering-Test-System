import cv2
from tensorflow import keras
from numpy import asarray, expand_dims

facenet_dir = "./models/facenet_keras.h5"


def facenet_preprocess_image(image):
    image = cv2.resize(image, (160, 160))

    pixels = asarray(image)
    pixels = pixels.astype('float32')

    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std

    samples = expand_dims(pixels, axis=0)

    return samples


def facenet_create_embeddings(loader, save_path, model_path=facenet_dir):
    model = keras.models.load_model(model_path)
    save_path = save_path.replace("\\", "/")

    with open(save_path, "w") as file:
        for image, image_path in loader.next_image():
            samples = facenet_preprocess_image(image)
            result = model.predict(samples)

            embedding = str(result[0])
            embedding = embedding.replace("\n", "")
            embedding = embedding.replace("[", "")
            embedding = embedding.replace("]", "")

            result_string = f"{image_path}\t{embedding}\n"
            file.write(result_string)
