import os

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from mAP import mean_average_precision


def main():
    category_mapping = {"0": "person"}

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="best.pt",
        confidence_threshold=0.5,
        category_mapping=category_mapping,
        device="cuda:0",
    )

    detections_list = []
    counter = 0
    for file in sorted(os.listdir("dataset/test_images"))[:10]:
        result = get_sliced_prediction(
            "dataset/test_images/" + file,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
        )
        result.export_visuals(
            export_dir="demo_data/",
            file_name=file,
            hide_labels=True,
            hide_conf=True,
            rect_th=2,
        )
        object_prediction_list = result.to_coco_annotations()
        for bbox in object_prediction_list:
            curr_detection = [counter, 0, bbox["score"]] + bbox["bbox"]
            detections_list.append(curr_detection)
        counter += 1

    counter = 0
    ground_truth_detections_list = []
    for file in sorted(os.listdir("dataset/test_labels"))[:10]:
        with open(os.path.join("dataset/test_labels", file), "r") as orig:
            boxes = orig.readlines()
            for bbox in boxes:
                curr = [counter, 0, None]
                for coords in bbox.split()[1:]:
                    curr.append(int(coords))
                ground_truth_detections_list.append(curr)
        counter += 1

    print(mean_average_precision(detections_list, ground_truth_detections_list))


main()
