import cv2
import dlib

import numpy as np

from src.extraction.align_utils import distance, rotate_point, is_between, cosine_formula, LANDMARKS_PREDICTOR_PATH, \
    INNER_EYES_AND_BOTTOM_LIP, OUTER_EYES_AND_NOSE, MINMAX_TEMPLATE


# Abstract class for face normalizing
class FaceNormalizer:
    Name = "Default Aligner Name"

    def normalize_face(self, image, face):
        pass


class MappingAligner(FaceNormalizer):
    Name = "Dlib Mapping Aligner"

    def __init__(self, indices=OUTER_EYES_AND_NOSE, detector_path=LANDMARKS_PREDICTOR_PATH, target_size=(224, 224)):
        self.Predictor = dlib.shape_predictor(detector_path)
        self.Indices = indices
        self.TargetSize = target_size

    def normalize_face(self, image, face):
        x, y, w, h = face["box"]
        bounding_box = dlib.rectangle(x, y, x + w, y + h)

        points = self.Predictor(image, bounding_box)
        landmarks = list(map(lambda p: (p.x, p.y), points.parts()))
        landmarks = np.float32(landmarks)

        landmark_indices = np.array(self.Indices)
        desired_points = MINMAX_TEMPLATE[landmark_indices] * np.array(self.TargetSize, dtype="float32")

        T = cv2.getAffineTransform(landmarks[landmark_indices], desired_points)
        image = cv2.warpAffine(image, T, self.TargetSize)

        return image


class EyesNoseAligner(FaceNormalizer):
    Name = "Eyes Nose Aligner"

    def normalize_face(self, image, face):
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

        M = cv2.getRotationMatrix2D((nose[0], nose[1]), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        image = image[y:y2, x:x2]

        return image


class EyesOnlyAligner(FaceNormalizer):
    Name = "Eyes Only Aligner"

    def __init__(self, left_eye=(0.3, 0.3)):
        self.LeftEyeDesiredLocation = left_eye

    def normalize_face(self, image, face):
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


class FaceCropperVGG(FaceNormalizer):
    Name = "VGG-Face Cropping"

    def __init__(self, margin=20, target_size=(224, 224)):
        self.Marging = margin
        self.TargetSize = target_size

    def normalize_face(self, image, face):
        img_h, img_w, _ = image.shape

        (x, y, w, h) = face["box"]
        margin = int(min(w, h) * self.Marging / 100)

        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin

        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h

        image = image[y_a: y_b, x_a: x_b]

        image = cv2.resize(image, self.TargetSize, interpolation=cv2.INTER_AREA)
        image = np.array(image)

        return image
