import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_bbox(img_path, label_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация в RGB для matplotlib
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()
    if lines:
        print(lines)
        print(label_path)
        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])

            x_center, y_center, bw, bh = parts[1], parts[2], parts[3], parts[4]
            x1 = int((x_center - bw / 2))
            y1 = int((y_center - bh / 2))
            x2 = int((x_center + bw / 2))
            y2 = int((y_center + bh / 2))
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.imshow(img)
        plt.axis("off")  # Скрыть оси
        plt.show()


def main():
    image_dir = "/home/gleb/learning/Detections/dataset/test_images"
    label_dir = "/home/gleb/learning/Detections/dataset/test_labels"

    for img_file in os.listdir(image_dir):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

            if os.path.exists(label_path):
                plot_bbox(img_path, label_path)


main()
