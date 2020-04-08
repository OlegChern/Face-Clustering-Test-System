from mtcnn import MTCNN
import cv2
import os


def extract_faces_mtcnn(loader, save_path):
    detector = MTCNN()
    save_path = save_path.replace("\\", "/")

    for image, image_path in loader.next_image():
        faces = detector.detect_faces(image)

        sep_pos = image_path.rfind("/") + 1
        dot_pos = image_path.rfind('.')
        image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
        save_dir = save_path + "/" + image_name

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for idx, face in enumerate(faces):
            x1, y1, width, height = face["box"]
            x2 = x1 + width
            y2 = y1 + height
            face_image = image[y1:y2, x1:x2]

            face_name = f"{image_name}_face_{idx}.jpg"
            face_path = save_dir + "/" + face_name

            cv2.imwrite(face_path, face_image)
