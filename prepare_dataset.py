import cv2
import os
import numpy as np
import csv


dictionary = {}

def create_mask(img_shape, bboxes):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1  # 1 = объект
    return mask


def process_image(img_path, annotations, output_dir, overlap = 50, crop_size = 640):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = create_mask((h, w), annotations)  # Создаем маску для всего изображения

    for y in range(0, h - crop_size + 1, 640 - overlap):
        for x in range(0, w - crop_size + 1, 640 - overlap):
            # Кроп изображения и маски
            img_crop = img[y:y + crop_size, x:x + crop_size]
            mask_crop = mask[y:y + crop_size, x:x + crop_size]

            # Сохранение кропа
            crop_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{x}_{y}"
            cv2.imwrite(f"{output_dir}/images/{crop_name}.jpg", img_crop)

            # Конвертация маски в YOLO-формат (полигоны)
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(f"{output_dir}/labels/{crop_name}.txt", "w") as f:
                for cnt in contours:
                    if cv2.contourArea(cnt) > 10:  # Игнорируем мелкие артефакты
                        cnt_norm = cnt.squeeze().astype(np.float32)
                        cnt_norm[:, 0] /= crop_size  # Нормализация X
                        cnt_norm[:, 1] /= crop_size  # Нормализация Y
                        f.write(f"0 " + " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in cnt_norm]) + "\n")


def main():
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)

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


    for image in os.listdir("/home/gleb/learning/Detections/dataset/train/"):
        if dictionary.get(image) is not None:
            process_image("/home/gleb/learning/Detections/dataset/train/" + image, dictionary[image], "dataset", overlap=100)





main()
