from mtcnn import MTCNN
from src.image_processing.utils import find_euclidean_distance
from PIL import Image

import cv2
import os
import numpy as np
import math


def extract_faces_mtcnn(loader, save_path, align=True):
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

            if align:
                image = align_face_eyes_nose(image, face["keypoints"], face["box"])

            face_name = f"{image_name}_face_{idx}.jpg"
            face_path = save_dir + "/" + face_name

            cv2.imwrite(face_path, face_image)


def align_face_eyes(image, keypoints):
    left_eye = np.array(keypoints["left_eye"])
    right_eye = np.array(keypoints["right_eye"])

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


def align_face_eyes_nose(image, keypoints, rectangle):
    left_eye = np.array(keypoints["left_eye"])
    right_eye = np.array(keypoints["right_eye"])
    nose = np.array(keypoints["nose"])

    x, y, w, h = rectangle
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
