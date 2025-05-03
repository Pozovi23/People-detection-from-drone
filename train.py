from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time


def sliding_window_inference(image_path, model, window_size=(640, 480), stride=320):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    h, w = img.shape[:2]
    results_all = []

    for y in range(0, h - window_size[1] + 1, stride):
        for x in range(0, w - window_size[0] + 1, stride):
            crop = img[y:y + window_size[1], x:x + window_size[0]]
            results = model.predict(crop, verbose=False)  # Детекция на кропе (YOLOv8 использует .predict())

            # Получаем боксы в формате xyxy
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()

            # Корректируем координаты относительно исходного изображения
            for box, conf, cl in zip(boxes, confs, cls):
                xmin, ymin, xmax, ymax = box
                xmin += x
                ymin += y
                xmax += x
                ymax += y
                results_all.append([xmin, ymin, xmax, ymax, conf, cl])

    # Фильтрация дубликатов (NMS)
    if len(results_all) > 0:
        results_all = torch.tensor(results_all)
        keep = torch.ops.torchvision.nms(
            results_all[:, :4],
            results_all[:, 4],
            iou_threshold=0.5  # Можно настроить
        )
        final_results = results_all[keep]
    else:
        final_results = torch.empty((0, 6))  # Пустой тензор, если ничего не найдено

    return final_results


def main():
    # Загрузка модели YOLOv8
    model = YOLO("yolov8n.pt")  # Автоматически скачает модель, если её нет

    # Пример использования
    start_time = time.time()
    detections = sliding_window_inference("/home/gleb/learning/parallels/bombardiro.jpg", model)

    # Визуализация
    img = cv2.imread("/home/gleb/learning/parallels/bombardiro.jpg")
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, cls = detection.tolist()
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.putText(img, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("output.jpg", img)
    print(f"Inference time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()