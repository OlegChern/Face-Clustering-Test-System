import math

import numpy as np
from PIL import Image
from enum import Enum, auto

from src.clustering.clustering_utils import find_euclidean_distance


class AlignType(Enum):
    EYES_NOSE, EYES_ONLY, NONE = auto(), auto(), auto()


def align_face_eyes(image, face):
    left_eye = np.array(face["keypoints"]["left_eye"])
    right_eye = np.array(face["keypoints"]["right_eye"])

    left_x = left_eye[0]
    left_y = left_eye[1]

    right_x = right_eye[0]
    right_y = right_eye[1]

    if left_y > right_y:
        point_third = np.asarray((right_x, left_y))
        rotation = -1
    else:
        point_third = np.asarray((left_x, right_y))
        rotation = 1

    a = find_euclidean_distance(left_eye, point_third)
    b = find_euclidean_distance(right_eye, point_third)
    c = find_euclidean_distance(right_eye, left_eye)

    angle_cos = (c * c + b * b - a * a) / (2 * b * c)
    angle = np.arccos(angle_cos)
    angle = (180 * angle) / math.pi

    if rotation == -1:
        angle = 90 - angle

    image = Image.fromarray(image)
    image = np.asarray(image.rotate(rotation * angle))

    return image


def align_face_eyes_nose(image, face):
    left_eye = np.array(face["keypoints"]["left_eye"])
    right_eye = np.array(face["keypoints"]["right_eye"])
    nose = np.array(face["keypoints"]["nose"])

    x, y, w, h = face["box"]
    forehead_center = np.array(((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2))
    upper_center = np.array((x + (w // 2), y))

    length_line1 = find_euclidean_distance(forehead_center, nose)
    length_line2 = find_euclidean_distance(upper_center, nose)
    length_line3 = find_euclidean_distance(upper_center, forehead_center)

    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    angle = np.arccos(cos_a)

    def rotate_point(origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def is_between(point1, point2, point3, extra_point):
        c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (
                extra_point[0] - point1[0])
        c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (
                extra_point[0] - point2[0])
        c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (
                extra_point[0] - point3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

    rotated_point = rotate_point(nose, forehead_center, angle)
    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

    if is_between(nose, forehead_center, forehead_center, rotated_point):
        rotation = -1
    else:
        rotation = 1

    angle = (180 * angle) / math.pi

    if rotation == -1:
        angle = 90 - angle

    image = Image.fromarray(image)
    image = np.array(image.rotate(rotation * angle))

    return image
