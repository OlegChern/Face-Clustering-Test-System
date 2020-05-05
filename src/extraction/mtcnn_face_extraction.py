from mtcnn import MTCNN

from timeit import default_timer

import cv2
import os


class FaceExtractorMTCNN:

    def __init__(self):
        self.AlignmentTime = 0
        self.Detector = MTCNN()

    def extract_faces(self, loader, save_path, aligner=None):
        self.AlignmentTime = 0

        for image, image_path in loader.next_image():
            faces = self.Detector.detect_faces(image)

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
            save_dir = save_path + "/" + image_name

            for idx, face in enumerate(faces):
                if aligner is not None:
                    start_time = default_timer()
                    face_image = aligner.align_face(image, face)
                    end_time = default_timer()

                    self.AlignmentTime += (end_time - start_time)
                else:
                    x1, y1, width, height = face["box"]
                    x2 = x1 + width
                    y2 = y1 + height
                    face_image = image[y1:y2, x1:x2]

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                face_name = f"{image_name}_face_{idx}.jpg"
                face_path = save_dir + "/" + face_name

                cv2.imwrite(face_path, face_image)