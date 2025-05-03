import cv2
import os
import numpy as np
import csv
import random
import shutil


dictionary = {}

def create_mask(img_shape, bboxes):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1  # 1 = объект
    return mask


def process_image(img_path, annotations, output_dir, overlap=50, crop_size=640):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = create_mask((h, w), annotations)

    for y in range(0, h - crop_size + 1, crop_size - overlap):
        for x in range(0, w - crop_size + 1, crop_size - overlap):
            img_crop = img[y:y + crop_size, x:x + crop_size]
            mask_crop = mask[y:y + crop_size, x:x + crop_size]

            crop_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{x}_{y}"
            cv2.imwrite(f"{output_dir}/raw_images/{crop_name}.jpg", img_crop)

            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            with open(f"{output_dir}/raw_labels/{crop_name}.txt", "w") as f:
                for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w > 50 and h > 50:
                            x_center = (x + w / 2) / crop_size
                            y_center = (y + h / 2) / crop_size
                            width = w / crop_size
                            height = h / crop_size

                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def prepare_test_images_and_labels(test_list, all_imgs_path: str):
    test_imgs_path = "dataset/test_images"
    test_labels_path = "dataset/test_labels"

    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    for image in test_list:
        shutil.copy(os.path.join(all_imgs_path, image), os.path.join(test_imgs_path, image))
        with open(os.path.join(test_labels_path, image)[:-4] + ".txt", "w") as f:
            if dictionary.get(image) is not None:
                for bbox in dictionary.get(image):
                    x1, y1, x2, y2 = bbox
                    f.write(f"0 {x1} {y1} {x2} {y2}\n")


def read_annotations():
    with open("dataset/_annotations.csv", "r") as f:
        reader = csv.reader(f)
        lines = f.readlines()

    for line in lines:
        if "train" in line:
            curr_line = line.strip().split(",")
            if dictionary.get(curr_line[0]) is None:
                dictionary[curr_line[0]] = []
            coords = [int(st) for st in curr_line[4:]]
            dictionary[curr_line[0]].append(coords)


def process_all_images():
    image_dir = "/home/gleb/learning/Detections/dataset/raw_images"
    label_dir = "/home/gleb/learning/Detections/dataset/raw_labels"

    labeled_imgs = []
    NOT_labeled_imgs = []

    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()

                if lines:
                    labeled_imgs.append(img_file)
                else:
                    NOT_labeled_imgs.append(img_file)

    return labeled_imgs, NOT_labeled_imgs


def prepare_train_val_images_and_labels(train_val_list, all_imgs_path):
    for image in train_val_list:
        if dictionary.get(image) is not None:
            process_image(os.path.join(all_imgs_path, image), dictionary[image], "dataset", overlap=100)


def train_val_split():
    labeled_imgs, NOT_labeled_imgs = process_all_images()

    random.shuffle(NOT_labeled_imgs)

    NOT_labeled_imgs = NOT_labeled_imgs[:len(labeled_imgs) * 5]

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    train_list = labeled_imgs[: int(0.85 * len(labeled_imgs))] + NOT_labeled_imgs[: int(0.85 * len(NOT_labeled_imgs))]
    val_list = labeled_imgs[int(0.85 * len(labeled_imgs)) : ] + NOT_labeled_imgs[int(0.85 * len(NOT_labeled_imgs)) : ]

    for image in train_list:
        shutil.copy(os.path.join("dataset/raw_images", image), os.path.join("dataset/images/train", image))
        shutil.copy(os.path.join("dataset/raw_labels", image)[:-4] + ".txt", os.path.join("dataset/labels/train", image)[:-4] + ".txt")
        os.remove(os.path.join("dataset/raw_images", image))
        os.remove(os.path.join("dataset/raw_labels", image)[:-4] + ".txt")

    for image in val_list:
        shutil.copy(os.path.join("dataset/raw_images", image), os.path.join("dataset/images/val", image))
        shutil.copy(os.path.join("dataset/raw_labels", image)[:-4] + ".txt", os.path.join("dataset/labels/val", image)[:-4] + ".txt")
        os.remove(os.path.join("dataset/raw_images", image))
        os.remove(os.path.join("dataset/raw_labels", image)[:-4] + ".txt")


def main():
    os.makedirs("dataset/raw_images", exist_ok=True)
    os.makedirs("dataset/raw_labels", exist_ok=True)
    all_imgs_path = "/home/gleb/learning/Detections/dataset/all_imgs"

    img_list = os.listdir(all_imgs_path)
    random.shuffle(img_list)

    read_annotations()

    train_val_list = img_list[:int(0.85 * len(img_list))]
    test_list = img_list[int(0.85 * len(img_list)):]

    prepare_train_val_images_and_labels(train_val_list, all_imgs_path)

    prepare_test_images_and_labels(test_list, all_imgs_path)

    train_val_split()


main()