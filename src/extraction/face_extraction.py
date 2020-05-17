from mtcnn import MTCNN

from src.extraction.normalization_utils import LANDMARKS_PREDICTOR_PATH, RIGHT_EYE, LEFT_EYE, NOSE

from timeit import default_timer
from abc import abstractmethod
from tqdm import tqdm

import cv2
import os
import dlib
import numpy as np


class FaceExtractor:
    Name = "Default Name"
    ExtractedFaces = 0
    NormalizationTime = 0

    def __init__(self, single_face=False):
        self.OnlyOneFace = single_face

    def extract_faces(self, loader, save_path, aligner=None):
        self.NormalizationTime = 0
        self.ExtractedFaces = 0

        save_path = save_path + "/" + self.Name

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = save_path + "/" + aligner.Name if aligner is not None else "No normalization"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        aligner_name = aligner.Name if aligner is not None else "no normalization"
        progress_bar = tqdm(loader.next_image(), total=loader.get_total_images_number(),
                            desc=f"Extracting faces for {self.Name} with {aligner_name}:", leave=True)

        for image, image_path in progress_bar:
            faces = self.get_bounding_boxes(image)

            if self.OnlyOneFace:
                if len(faces) > 0:
                    faces = [max(faces, key=lambda f: f["box"][2] * f["box"][3])]

            sep_pos = image_path.rfind("/") + 1
            dot_pos = image_path.rfind('.')
            image_name = image_path[sep_pos:dot_pos].replace("\\", "/")

            for idx, face in enumerate(faces):
                self.ExtractedFaces += 1

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

                save_dir = save_path + "/" + image_name
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                face_name = f"{image_name}_face_{idx}.jpg"
                face_path = save_dir + "/" + face_name

                cv2.imwrite(face_path, face_image)

    @abstractmethod
    def get_bounding_boxes(self, image):
        ...


class FaceExtractorMTCNN(FaceExtractor):
    Name = "MTCNN Face Extractor"

    def __init__(self, single_face=False):
        self.NormalizationTime = 0
        self.Detector = MTCNN()
        super().__init__(single_face)

    def get_bounding_boxes(self, image):
        return self.Detector.detect_faces(image)


class FaceExtractorDlib(FaceExtractor):
    Name = "Dlib-based Face Extractor"

    def __init__(self, landmarks_predictor_path=LANDMARKS_PREDICTOR_PATH, single_face=False):
        self.NormalizationTime = 0
        self.Detector = dlib.get_frontal_face_detector()
        self.LandmarksPredictor = dlib.shape_predictor(landmarks_predictor_path)
        super().__init__(single_face)

    def get_bounding_boxes(self, image):
        faces = self.Detector(image, 1)
        boxes = list()

        for face in faces:
            x1 = face.left() if face.left() > 0 else 0
            y1 = face.top() if face.top() > 0 else 0
            width = face.width()
            height = face.height()

            box = self.get_face_dict(image, x1, y1, width, height)
            boxes.append(box)

        return boxes

    def get_face_dict(self, image, x1, y1, width, height):
        bounding_box = dlib.rectangle(x1, y1, x1 + width, y1 + height)
        points = self.LandmarksPredictor(image, bounding_box)
        points = list(map(lambda p: (p.x, p.y), points.parts()))
        points = np.float32(points)

        box = dict()
        box.update({"box": (x1, y1, width, height)})
        box.update({"keypoints": {
            "left_eye": points[LEFT_EYE].mean(axis=0).astype("int"),
            "right_eye": points[RIGHT_EYE].mean(axis=0).astype("int"),
            "nose": points[NOSE].mean(axis=0).astype("int")
        }})

        return box


class FaceExtractorLFW(FaceExtractorDlib):
    Name = "LFW Normalization"

    def get_bounding_boxes(self, image):
        return [self.get_face_dict(image, 0, 0, image.shape[1], image.shape[0])]
