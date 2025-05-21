from collections import Counter
import torch
import os
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import json


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def draw_boxes_with_ids(image, object_prediction_list):
    """Draw bounding boxes with unique IDs on the image"""
    image = cv2.imread(image) if isinstance(image, str) else image
    for idx, pred in enumerate(object_prediction_list):
        bbox = pred["bbox"]
        x, y, w, h = map(int, bbox)

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw ID text
        cv2.putText(image, f"ID:{idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


def mean_average_precision(pred_boxes, true_boxes, object_prediction_list, iou_threshold=0.5, num_classes=1):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[-1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[-1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[5], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        print("\nDetection analysis results:")
        print("ID | IoU    | Predicted Box [x,y,w,h]       | Ground Truth Box [x,y,w,h]")
        print("-" * 80)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[1:-1]),
                    torch.tensor(gt[1:5]),
                )
                if iou > 0.001:
                    # Find detection ID by matching coordinates and score
                    detection_id = None
                    for i, pred in enumerate(object_prediction_list):
                        if (np.allclose(pred['bbox'], detection[1:5], atol=1e-3) and
                                np.isclose(pred['score'], detection[-2], atol=1e-3)):
                            detection_id = i
                            break

                    if detection_id is not None:
                        print(f"{detection_id:2} | {iou} | {str(detection[1:-1]):30} | {gt[1:5]}")


def main():
    category_mapping = {"0": "person"}

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="best.pt",
        confidence_threshold=0,
        device="cuda:0",
    )

    detections_list = []
    counter = 0
    image_path = "dataset/test_images/train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.jpg"

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_metric='IOU',
        postprocess_type="NMS",
        postprocess_match_threshold=0.6,
    )

    object_prediction_list = result.to_coco_annotations()
    print(object_prediction_list)
    # Draw boxes with IDs and save
    image_with_ids = draw_boxes_with_ids(image_path, object_prediction_list)
    output_path = "demo_data/0_with_ids.png"
    os.makedirs("demo_data", exist_ok=True)
    cv2.imwrite(output_path, image_with_ids)
    print(f"\nSaved image with box IDs to {output_path}")

    for bbox in object_prediction_list:
        curr_detection = [counter] + bbox["bbox"] + [bbox["score"], 0]
        detections_list.append(curr_detection)
    counter += 1

    counter = 0
    ground_truth_detections_list = []
    label_path = os.path.join("dataset/test_labels", "train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.txt")
    with open(label_path, "r") as orig:
        boxes = orig.readlines()
        for bbox in boxes:
            curr = [counter]
            for coords in bbox.split()[1:]:
                curr.append(float(coords))
            curr.append(0)
            ground_truth_detections_list.append(curr)
    counter += 1

    mas = []
    for pred in result.object_prediction_list:
        curr_dict = {
            "image_id": 0,
            "category_id": 0,
            "bbox": pred.bbox.to_coco_bbox(),
            "score": pred.score.value,
        }
        print(curr_dict, pred.bbox.to_xywh())
        mas.append(curr_dict)

    with open('predictions.json', 'w') as f:
        json.dump(mas, f)
    print("\nStarting mAP calculation...")
    mean_average_precision(detections_list, ground_truth_detections_list, object_prediction_list)


if __name__ == "__main__":
    main()