import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def plot_bbox(img_path, label_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация в RGB для matplotlib
    h, w = img.shape[:2]

    # Чтение аннотаций из файла
    with open(label_path, "r") as f:
        lines = f.readlines()
    if lines:
        print(label_path)
        # Отрисовка bounding box
        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])

            x_center, y_center, bw, bh = parts[1], parts[2], parts[3], parts[4]
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Синий прямоугольник

        plt.imshow(img)
        plt.axis('off')  # Скрыть оси
        plt.show()


def main():
    image_dir = "/home/gleb/learning/Detections/dataset/train_images"
    label_dir = "/home/gleb/learning/Detections/dataset/train_labels"

    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                plot_bbox(img_path, label_path)


main()