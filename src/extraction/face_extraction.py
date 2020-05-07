from mtcnn import MTCNN

from src.extraction.align_utils import LANDMARKS_PREDICTOR_PATH, RIGHT_EYE, LEFT_EYE, NOSE

from timeit import default_timer

import cv2
import os
import dlib


class FaceExtractor:
    NormalizationTime = 0

    def extract_faces(self, loader, save_path, aligner=None):
        pass


class FaceExtractorMTCNN(FaceExtractor):

    def __init__(self):
        self.NormalizationTime = 0
        self.Detector = MTCNN()

    def extract_faces(self, loader, save_path, aligner=None):
        self.NormalizationTime = 0

        for image, image_path in loader.next_image():
            faces = self.Detector.detect_faces(image)

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
            save_dir = save_path + "/" + image_name

            for idx, face in enumerate(faces):
                if aligner is not None:
                    start_time = default_timer()
                    face_image = aligner.normalize_face(image, face)
                    end_time = default_timer()

                    self.NormalizationTime += (end_time - start_time)
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


class FaceExtractorDlib(FaceExtractor):

    def __init__(self, landmarks_predictor_path=LANDMARKS_PREDICTOR_PATH):
        self.AlignmentTime = 0
        self.Detector = dlib.get_frontal_face_detector()
        self.LandmarksPredictor = dlib.shape_predictor(landmarks_predictor_path)

    def extract_faces(self, loader, save_path, aligner=None):
        self.AlignmentTime = 0

        for image, image_path in loader.next_image():
            faces = self.Detector(image, 1)

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")
            save_dir = save_path + "/" + image_name

            for idx, face in enumerate(faces):
                x1 = face.left() if face.left() > 0 else 0
                y1 = face.top() if face.top() > 0 else 0
                width = face.width()
                height = face.height()

                if aligner is not None:
                    bounding_box = dlib.rectangle(x1, y1, x1 + width, y1 + height)
                    points = self.LandmarksPredictor(image, bounding_box)

                    face = dict()
                    face.update({"box": (x1, y1, width, height)})
                    face.update({"keypoints": {
                        "left_eye": points[LEFT_EYE].mean(axis=0).astype("int"),
                        "right_eye": points[RIGHT_EYE].mean(axis=0).astype("int"),
                        "nose": points[NOSE].mean(axis=0).astype("int")
                    }})

                    start_time = default_timer()
                    face_image = aligner.normalize_face(image, face)
                    end_time = default_timer()

                    self.AlignmentTime += (end_time - start_time)
                else:
                    x2 = x1 + width
                    y2 = y1 + height
                    face_image = image[y1:y2, x1:x2]

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                face_name = f"{image_name}_face_{idx}.jpg"
                face_path = save_dir + "/" + face_name

                cv2.imwrite(face_path, face_image)
