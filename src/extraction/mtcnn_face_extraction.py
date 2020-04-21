from mtcnn import MTCNN
from src.image_processing.utils import find_euclidean_distance
from PIL import Image

import cv2
import os
import numpy as np
import math


def extract_faces_mtcnn(loader, save_path, align=False):
    detector = MTCNN()
    save_path = save_path.replace("\\", "/")

    eye_detector = get_eye_detector() if align else None

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
                image = align_face_opencv(image, eye_detector)

            face_name = f"{image_name}_face_{idx}.jpg"
            face_path = save_dir + "/" + face_name

            cv2.imwrite(face_path, face_image)


def align_face_opencv(image, eye_detector):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = np.asarray(eye_detector.detectMultiScale(grey_image))

    if len(eyes) >= 2:
        eyes = eyes[np.argsort(eyes[:, 2])]

        eye_first = eyes[-1]
        eye_second = eyes[-2]

        if eye_first[0] < eye_second[0]:
            left = eye_first
            right = eye_second
        else:
            left = eye_second
            right = eye_first

        left_center = np.asarray((int(left[0] + (left[2] / 2)), int(left[1] + (left[3] / 2))))
        left_x = left_center[0]
        left_y = left_center[1]

        right_center = np.asarray((int(right[0] + (right[2] / 2)), int(right[1] + (right[3] / 2))))
        right_x = right_center[0]
        right_y = right_center[1]

        if left_y > right_y:
            point_third = np.asarray((right_x, left_y))
            rotation = -1
        else:
            point_third = np.asarray((left_x, right_y))
            rotation = 1

        a = find_euclidean_distance(left_center, point_third)
        b = find_euclidean_distance(right_center, point_third)
        c = find_euclidean_distance(right_center, left_center)

        angle_cos = (c * c + b * b - a * a) / (2 * b * c)
        angle = np.arccos(angle_cos)
        angle = (180 * angle) / math.pi

        if rotation == -1:
            angle = 90 - angle

        image = Image.fromarray(image)
        image = np.asarray(image.rotate(rotation * angle))

    return image


def get_eye_detector():
    opencv_home = cv2.__file__

    last_sep = opencv_home.rfind(os.path.sep)
    eye_detector_path = os.path.join(opencv_home[:last_sep], "data", "haarcascade_eye.xml")

    return cv2.CascadeClassifier(eye_detector_path)
