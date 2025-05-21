import cv2
import torch
import time
from ultralytics import YOLO
import queue
import threading


def predict(weights, img_queue, output_queue, stop_event):
    local_model = YOLO(weights, task='detect')

    while not stop_event.is_set():
        try:
            img, x, y = img_queue.get(timeout=0.1)
            results = local_model.predict(img,  device="intel:сpu", conf=0.25, iou=0.6)
            output_queue.put((results, x, y))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error: {e}")
            break


def sliding_window_inference(image_path, weights, window_size=640, overlap=0.2, number_of_threads=8):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    results_all = []

    stride = int(window_size * (1 - overlap))  # 512 при overlap=0.2

    pad_w = (stride - (w - window_size) % stride) % stride
    pad_h = (stride - (h - window_size) % stride) % stride

    if pad_w > 0 or pad_h > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    stop_event = threading.Event()
    output_queue = queue.Queue()
    frame_queue = queue.Queue()
    frames_cnt = 0
    for y in range(0, h + pad_h - window_size + 1, stride):
        print(y)
        for x in range(0, w + pad_w - window_size + 1, stride):
            frames_cnt+=1
            frame_queue.put((img[y:y + window_size, x:x + window_size], x, y))


    yolo_threads = []

    for i in range(number_of_threads):
        yolo_threads += [
            threading.Thread(
                target=predict, args=(weights, frame_queue, output_queue, stop_event)
            )
        ]
        yolo_threads[i].start()

    results_all_not_done = []

    start_time = time.time()
    cnt = 0
    while (
            frames_cnt != cnt
    ):
        results_all_not_done.append(output_queue.get())
        cnt += 1

    stop_event.set()

    for elem in results_all_not_done:
        results, x, y = elem
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().copy()
                    xyxy[0] += x
                    xyxy[1] += y
                    xyxy[2] += x
                    xyxy[3] += y
                    conf = box.conf.item()
                    cls = box.cls.item()
                    results_all.append([*xyxy, conf, cls])


    print(len(results_all))
    if len(results_all) > 0:
        results_all = torch.tensor(results_all)
        keep = torch.ops.torchvision.nms(
            results_all[:, :4],
            results_all[:, 4],
            iou_threshold=0.6
        )
        final_results = results_all[keep]
    else:
        final_results = torch.empty((0, 6))

    return final_results, start_time


def inference_openvino(weights, img_path, out_path):
    detections, start_time = sliding_window_inference(img_path, weights)
    img = cv2.imread(img_path)
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, cls = detection.tolist()
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    print(time.time() - start_time)


def openvino_export(model_path):
    model = YOLO(model_path)
    model.export(format="openvino", device='cpu')



inference_openvino("/home/gleb/learning/Detections/best_openvino_model/", "dataset/test_images/train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.jpg", "readme_dir/out_openvino_parallel.png")