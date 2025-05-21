from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import torch
import time
from ultralytics import YOLO


def inference_sahi(filepath_src, file_dir_out, filename_out, weigths_path, conf_th = 0.25, device = "cuda:0", save=False, hide_labels=False, hide_conf=False):
    category_mapping = {"0": "person"}
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weigths_path,
        confidence_threshold=conf_th,
        category_mapping=category_mapping,
        device=device,
    )
    start_timer = time.time()
    result = get_sliced_prediction(
        filepath_src,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_metric='IOU',
        postprocess_type="NMS",
        postprocess_match_threshold=0.6,

    )
    print(time.time() - start_timer)
    if save:
        result.export_visuals(
            export_dir=file_dir_out,
            file_name=filename_out,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            rect_th=2,
        )


def sliding_window_inference(image_path, model, window_size=640, overlap=0.2):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    results_all = []

    stride = int(window_size * (1 - overlap))

    pad_w = (stride - (w - window_size) % stride) % stride
    pad_h = (stride - (h - window_size) % stride) % stride

    if pad_w > 0 or pad_h > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    for y in range(0, h + pad_h - window_size + 1, stride):
        print(y)
        for x in range(0, w + pad_w - window_size + 1, stride):
            crop = img[y:y + window_size, x:x + window_size]
            results = model.predict(crop, device="intel:cpu", conf=0.25, iou=0.6)

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

    return final_results


def inference_openvino(weights, img_path, out_path):
    model = YOLO(weights, task='detect')
    start_time = time.time()
    detections = sliding_window_inference(img_path, model)
    img = cv2.imread(img_path)
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, cls = detection.tolist()
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    print(time.time() - start_time)


def openvino_export(model_path):
    model = YOLO(model_path)
    model.export(format="openvino", device='cpu', batch = 4, dynamic=True, half=False)

# openvino_export("best.pt")

# inference_sahi("dataset/test_images/train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.jpg", "./", "readme_dir/out_sahi.png", "best.pt", save=True, device='cuda:0')

inference_openvino("/home/gleb/learning/Detections/best_openvino_model/", "dataset/test_images/train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.jpg", "readme_dir/out_openvino.png")