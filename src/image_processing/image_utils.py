import os
import shutil


def extract_person_name(labeled_image):
    image_path, label = labeled_image

    sep_pos = image_path.rfind("/") + 1
    dot_pos = image_path.rfind('.')
    image_name = image_path[sep_pos:dot_pos].replace("\\", "/")

    name_parts = image_name.split("_")
    if len(name_parts) > 2:
        person_name = f"{name_parts[0]}_{name_parts[1]}"
    else:
        person_name = name_parts[0]

    return image_path, person_name, label


def sort_images(labeled_names, save_path):
    for image_path, label in labeled_names:
        label_dir = os.path.join(save_path, f"person_{label}")

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        sep_pos = image_path.rfind("/") + 1
        image_name = image_path[sep_pos:]
        dest_path = os.path.join(label_dir, image_name)

        shutil.copyfile(image_path, dest_path)