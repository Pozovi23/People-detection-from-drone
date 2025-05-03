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

            # Если формат YOLO (normalized xywh)
            if len(parts) == 5:  # Класс + x_center, y_center, width, height
                x_center, y_center, bw, bh = parts[1], parts[2], parts[3], parts[4]
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Синий прямоугольник

            # Если формат YOLO сегментации (полигоны)
            elif len(parts) > 5:  # Класс + полигон (x1,y1,x2,y2,...)
                polygon = np.array(parts[1:]).reshape(-1, 2) * np.array([w, h])
                polygon = polygon.astype(np.int32)
                x1, y1 = np.min(polygon[:, 0]), np.min(polygon[:, 1])
                x2, y2 = np.max(polygon[:, 0]), np.max(polygon[:, 1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленый прямоугольник

        plt.imshow(img)
        plt.axis('off')  # Скрыть оси
        plt.show()


def main():
    image_dir = "/home/gleb/learning/Detections/dataset/images"
    label_dir = "/home/gleb/learning/Detections/dataset/labels"

    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                plot_bbox(img_path, label_path)


def count():
    image_dir = "/home/gleb/learning/Detections/dataset/images"
    label_dir = "/home/gleb/learning/Detections/dataset/labels"
    i = 0
    j = 0
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if os.path.exists(label_path):
                j += 1
                with open(label_path, "r") as f:
                    lines = f.readlines()

                if lines:
                    i += 1
    print(i, j)

main()