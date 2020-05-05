import cv2
import dlib

import numpy as np
from PIL import Image
from enum import Enum, auto

from src.extraction.align_utils import distance, rotate_point, is_between, cosine_formula, LANDMARKS_PREDICTOR_PATH, \
    INNER_EYES_AND_BOTTOM_LIP, OUTER_EYES_AND_NOSE, MINMAX_TEMPLATE


# Abstract class for face aligners
class FaceAligner:
    Name = "DefaultAlignerName"

    def align_face(self, image, face):
        pass


class MappingAligner(FaceAligner):
    Name = "Mapping_Aligner"

    def __init__(self, indices=INNER_EYES_AND_BOTTOM_LIP, detector_path=LANDMARKS_PREDICTOR_PATH):
        self.Predictor = dlib.shape_predictor(detector_path)
        self.Indices = indices

    def align_face(self, image, face):
        x, y, w, h = face["box"]
        bounding_box = dlib.rectangle(x, y, x + w, y + h)

        points = self.Predictor(image, bounding_box)
        landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
        landmarks = np.float32(landmarks)

        landmark_indices = np.array(self.Indices)

        desired_points = MINMAX_TEMPLATE[landmark_indices] * np.array([w, h], dtype="float32")

        T = cv2.getAffineTransform(landmarks[landmark_indices], desired_points)
        image = cv2.warpAffine(image, T, (w, h))

        return image


class EyesNoseAligner(FaceAligner):
    Name = "Eyes&Nose_Aligner"

    def align_face(self, image, face):
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        nose = face["keypoints"]["nose"]

        x, y, w, h = face["box"]
        x2 = x + w
        y2 = y + h

        center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        center_pred = (x + w // 2, y)

        length_line1 = distance(center_of_forehead, nose)
        length_line2 = distance(center_pred, nose)
        length_line3 = distance(center_pred, center_of_forehead)

        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)

        rotated_point = rotate_point(nose, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

        if is_between(nose, center_of_forehead, center_pred, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)

        M = cv2.getRotationMatrix2D(nose, angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        image = image[y:y2, x:x2]

        return image


class EyesOnlyAligner(FaceAligner):
    Name = "Eyes_Only_Aligner"

    def __init__(self, left_eye=(0.25, 0.25)):
        self.LeftEyeDesiredLocation = left_eye

    def align_face(self, image, face):
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]

        _, _, w, h = face["box"]
        target_size = (w, h)

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_right_eye_x = 1.0 - self.LeftEyeDesiredLocation[0]

        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = (desired_right_eye_x - self.LeftEyeDesiredLocation[0])
        desired_dist *= target_size[0]
        scale = desired_dist / dist

        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = target_size[0] * 0.5
        tY = target_size[1] * self.LeftEyeDesiredLocation[1]

        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        image = cv2.warpAffine(image, M, target_size, flags=cv2.INTER_CUBIC)

        return image
